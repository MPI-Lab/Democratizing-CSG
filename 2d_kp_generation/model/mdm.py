import numpy as np
import torch
import clip
import torch.nn as nn
import warnings
from transformers import AutoProcessor, Wav2Vec2Model
import torch.nn.functional as F
# proxies = {
#     'http': '127.0.0.1:7890',
#     'https': '127.0.0.1:7890',
# }

class MDM(nn.Module):
    def __init__(
        self,
        njoints,
        nfeats,
        num_actions,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        clip_dim=512,
        arch="trans_enc",
        emb_trans_dec=False,
        clip_version=None,
        cond_mode="no_cond",
        cond_mask_prob=0.0,
        speaker_id=False,
        speaker_num=-1,
        **kargs,
    ):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.clip_dim = clip_dim
        self.cond_mode = cond_mode
        self.cond_mask_prob = cond_mask_prob
        self.arch = arch
        self.input_feats = self.njoints * self.nfeats
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0

        self.input_process = InputProcess(self.input_feats + self.gru_emb_dim, self.latent_dim)

        if "audio_concat" in self.cond_mode:
            self.concat_to_hidden = nn.Sequential(nn.Linear(2*self.latent_dim, 4*self.latent_dim), nn.GELU(), nn.Linear(4*self.latent_dim, self.latent_dim))
            # self.concat_to_hidden = nn.Linear(2*self.latent_dim, self.latent_dim)

        if "initial_kp_pose_guider" in self.cond_mode:
            self.initial_kp_pose_guider = PoseGuider(self.input_feats, self.latent_dim)

        # if "long_video" in self.cond_mode:
        #     self.reference_kp_pose_guider = PoseGuider(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        self.speaker_id = speaker_id
        self.speaker_num = speaker_num
        if self.arch == "trans_enc":
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        elif self.arch == "trans_dec":
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

        elif self.arch == "gru":
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)

        else:
            raise ValueError("Please choose a valid architecture [trans_enc, trans_dec, gru]")

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.speaker_id:
            assert speaker_num > 0
            self.speaker_embedding = nn.Linear(self.speaker_num, self.latent_dim)

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print("Loading CLIP...")
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if "action" in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
            if 'audio' in self.cond_mode:
                # self.aud_m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", proxies=proxies)
                self.aud_m = Wav2Vec2Model.from_pretrained("./pretrained_weights/wav2vec2-base-960h")
                # set self.aud_m not trainable
                self.aud_m.eval()
                for param in self.aud_m.parameters():
                    param.requires_grad = False
                self.aud_dim = 768
                self.embed_audio = nn.Linear(self.aud_dim, self.latent_dim)
                self.audio_learnable = nn.Parameter(torch.randn(self.latent_dim), requires_grad=True)

            if 'transcription' in self.cond_mode or 'description' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print("Loading CLIP...")
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

            if 'description' in self.cond_mode:
                pass
            if 'transcription' in self.cond_mode:
                pass

        self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints, self.nfeats)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load("./pretrained_weights/clip/ViT-B-32.pt", device="cpu", jit=False)  # Must set jit=False for training
        clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = None
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)  # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device)  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def mask_cond(self, cond):
        bs = cond.shape[0]
        if self.training and self.cond_mask_prob > 0.0:
            target_shape = np.ones(len(cond.shape), dtype=int)
            target_shape[0] = bs
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(list(target_shape))
            # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def get_seq_mask(self, xseq, y, additional_emb_num):
        lengths = y["lengths"].to(xseq.device).view(-1, 1) + additional_emb_num  # [bs, 1]
        positions = torch.arange(xseq.size(0), device=xseq.device).view(1, -1)  # [1, seqlen+1]
        return positions >= lengths  # [seqlen+1, bs]



    def _audio_resize(self, hidden_state: torch.Tensor, output_len):
        """
        Resize the audio feature to the same length as the vertice
        Args:
            hidden_state (torch.Tensor): [batch_size, hidden_size, seq_len]
            input_fps (float): input fps
            output_fps (float): output fps
            output_len (int): output length
        """
        hidden_state = hidden_state.transpose(1,2)
        output_features = F.interpolate(hidden_state, size = output_len, align_corners=True, mode="linear")
        return output_features.transpose(2,1)

    def _audio_2_hidden(self, audio, audio_attention, length = None):
        """
        This function takes in an audio tensor and its corresponding attention mask, 
        and returns a hidden state tensor that represents the audio feature map. 
        The function first passes the audio tensor through an audio encoder to obtain the last hidden state. 
        It then resizes the hidden state to match the length of the input sequence, using the _audio_resize function. 
        Finally, the function passes the resized hidden state through the audio_feature_map layer of the denoiser to obtain the final hidden state tensor. 
        The output tensor has shape [batch_size, seq_len, latent_dim], where seq_len is the length of the input audio sequence and latent_dim is the dimensionality of the latent space.
        """
        with torch.no_grad():
            # hidden_state = self.aud_m(audio, attention_mask = audio_attention).last_hidden_state

            hidden_state = self.aud_m(audio).last_hidden_state

                        
        stride = 16000 // 50
        if length:
            stride = 16000 // 25
            hidden_state = self._audio_resize(
                hidden_state, 
                output_len = length
            )

        kernel_size = stride
        max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        audio_mask = max_pool(audio_attention.unsqueeze(1)).squeeze(1)[:,:hidden_state.size(1)]

        hidden_state = self.embed_audio(hidden_state) # hidden_state.shape = [batch_size, seq_len, latent_dim]
        return hidden_state, audio_mask




    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if self.speaker_id:
            emb += self.speaker_embedding(y['conditions']["speaker_id"])

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                enc_text = self.encode_text(y["text"])
                emb += self.embed_text(self.mask_cond(enc_text))

            if "action" in self.cond_mode:
                action_emb = self.embed_action(y["action"])
                emb += self.mask_cond(action_emb)

            if 'audio' in self.cond_mode:
                audio_processed = y["conditions"]["audio"]
                audio_attention = y["conditions"]["audio_attention"]

                if "audio_concat" in self.cond_mode:
                    audio_ft, audio_mask = self._audio_2_hidden(audio_processed, audio_attention, length = nframes)
                else:
                    audio_ft, audio_mask = self._audio_2_hidden(audio_processed, audio_attention)
                
                if 'audio_position_encoding' in self.cond_mode:
                    audio_ft = self.sequence_pos_encoder(audio_ft)
                
                if self.training and self.cond_mask_prob > 0.0:
                    mask = torch.bernoulli(torch.ones(bs, device=audio_ft.device) * self.cond_mask_prob).bool().view(-1, 1, 1)
                    audio_learnable_expanded = self.audio_learnable.unsqueeze(0).unsqueeze(0).expand(bs, audio_ft.size(1), audio_ft.size(2))
                    # use mask to determine whether to use self.audio_learnable or audio_ft
                    audio_ft = torch.where(mask, audio_learnable_expanded, audio_ft).permute(1, 0, 2)  # [seq_len, bs, d]
                elif not self.training and y.get("use_learnable_uncond", False):
                    audio_ft = self.audio_learnable.unsqueeze(0).unsqueeze(0).expand(bs, audio_ft.size(1), audio_ft.size(2)).permute(1, 0, 2) 
                else:
                    audio_ft = audio_ft.permute(1, 0, 2)

            if 'initial_kp' in self.cond_mode:
                enc_initial_kp = y["conditions"]["initial_kp"]

            if 'long_video' in self.cond_mode:
                # reference_kp = y["conditions"]["reference_kp"]
                initial_frames = y["conditions"]["initial_frames"]

            if 'description' in self.cond_mode:
                enc_description = self.encode_text(y["conditions"]["description"])
                emb_description = self.embed_text(self.mask_cond(enc_description))
                emb += emb_description
            if 'transcription' in self.cond_mode:
                enc_transcription = self.encode_text(y["transcription"])
                emb_transcription = self.embed_text(self.mask_cond(enc_transcription))
                emb_transcription += emb


            

        if self.arch == "gru":
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)  # [#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)  # [bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  # [bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  # [bs, d+joints*feat, 1, #frames]



        additional_emb_num = 0

        if 'initial_kp_pose_guider' in self.cond_mode:
            initial_kp_pose_guider = self.initial_kp_pose_guider(enc_initial_kp)
            x = x + initial_kp_pose_guider

        if 'initial_kp' in self.cond_mode and 'initial_kp_pose_guider' not in self.cond_mode:
            x = torch.cat((enc_initial_kp, x), axis=-1)
            additional_emb_num += 1

        if 'long_video' in self.cond_mode:
            # reference_kp_pose_guider = self.reference_kp_pose_guider(reference_kp)
            # x = x + reference_kp_pose_guider
            
            x = torch.cat((initial_frames, x), axis=-1)
            additional_emb_num += initial_frames.shape[-1]
    

        x = self.input_process(x)

        if "audio_concat" in self.cond_mode:
            if "initial_kp" in self.cond_mode and 'initial_kp_pose_guider' not in self.cond_mode:
                audio_ft = torch.cat((torch.zeros_like(audio_ft[:1]), audio_ft), axis=0)
            
            if "long_video" in self.cond_mode:
                audio_ft = torch.cat((torch.zeros_like(audio_ft[:initial_frames.shape[-1]]), audio_ft), axis=0)

            x = self.concat_to_hidden(torch.cat((x, audio_ft), axis=-1))

        if self.arch == "trans_enc":
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            additional_emb_num += 1
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=self.get_seq_mask(xseq, y, additional_emb_num))[additional_emb_num:]  # [seqlen, bs, d]

        elif self.arch == "trans_dec":
            if self.emb_trans_dec:
                # if 'transcription' in self.cond_mode or 'description' in self.cond_mode:
                #     if 'transcription' in self.cond_mode:
                #         xseq = torch.cat((emb_transcription, x), axis=0)
                #         additional_emb_num += 1
                #     if 'description' in self.cond_mode:
                #         xseq = torch.cat((emb_description, x), axis=0)
                #         additional_emb_num += 1
                # else:
                xseq = torch.cat((emb, x), axis=0)
                additional_emb_num += 1
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                oringinal_mask = self.get_seq_mask(xseq, y, additional_emb_num)
                tgt_key_padding_mask = oringinal_mask      
                # memory_key_padding_mask = oringinal_mask[:, additional_emb_num:]
                memory_key_padding_mask = (1-audio_mask).bool()
                output = self.seqTransDecoder(tgt=xseq, memory=audio_ft, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)[additional_emb_num:]
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == "gru":
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

class PoseGuider(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Sequential(nn.Linear(input_feats, latent_dim), nn.GELU(), nn.Linear(latent_dim, input_feats))

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        x = x.reshape(nframes, bs, njoints, nfeats)
        x = x.permute((1,2,3,0))
        return x




class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
