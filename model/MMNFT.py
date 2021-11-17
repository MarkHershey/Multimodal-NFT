"""Multi-Modal NFT"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from .utils import init_modules


class TextualInputModule(nn.Module):
    def __init__(
        self,
        vocab_size,
        glove_matrix,
        wordvec_dim=300,
        rnn_dim=512,
        module_dim=512,
        bidirectional=True,
    ):
        super(TextualInputModule, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim).from_pretrained(
            glove_matrix, freeze=False
        )
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(
            wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional
        )
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.text_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, texts, text_len: int):
        """
        Args:
            texts: [Tensor] (batch_size, max_text_len)
            text_len: [Tensor] (batch_size)
        return:
            texts representation [Tensor] (batch_size, module_dim)
        """
        texts_embedding = self.encoder_embed(texts)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(texts_embedding))

        # Ref: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        embed = nn.utils.rnn.pack_padded_sequence(
            input=embed,  # padded batch of variable length sequences
            lengths=text_len,  #  list of sequence lengths of each batch element
            batch_first=True,  # if True, the input is expected in B x T x * format.
            enforce_sorted=False,  # If False, the input will get sorted unconditionally
        )

        # self.encoder.flatten_parameters()
        output, (h_n, c_n) = self.encoder(embed)

        if self.bidirectional:
            h_n = torch.cat([h_n[0], h_n[1]], -1)

        h_n = self.text_dropout(h_n)

        return h_n


class StillVisualInputModule(nn.Module):
    def __init__(
        self,
        k_max_frame_level,
        k_max_clip_level,
        spl_resolution,
        vision_dim,
        module_dim=512,
    ):
        super(StillVisualInputModule, self).__init__()

        ...
        self.clip_level_motion_proj = nn.Linear(vision_dim, module_dim)
        self.video_level_motion_proj = nn.Linear(module_dim, module_dim)
        self.appearance_feat_proj = nn.Linear(vision_dim, module_dim)

        self.question_embedding_proj = nn.Linear(module_dim, module_dim)

        self.module_dim = module_dim
        self.activation = nn.ELU()

    def forward(self, appearance_video_feat, motion_video_feat):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        # print(">>> appearance_video_feat", appearance_video_feat.size())
        # print(">>> motion_video_feat", motion_video_feat.size())
        # print(">>> question_embedding", question_embedding.size())
        batch_size = appearance_video_feat.size(0)
        clip_level_crn_outputs = []
        ...
        return ...


class MotionVisualInputModule(nn.Module):
    def __init__(self, dim):
        super(MotionVisualInputModule, self).__init__()

        self.dim = dim
        self.activation = nn.ELU()


class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.t_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, textual_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        t_proj = self.t_proj(textual_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, t_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)

        return v_distill


class MMNFT(nn.Module):
    def __init__(
        self,
        vision_dim,
        module_dim,
        word_dim,
        k_max_frame_level,
        k_max_clip_level,
        spl_resolution,
        vocab,
    ):
        super(MMNFT, self).__init__()

        self.feature_aggregation = FeatureAggregation(module_dim)

        encoder_vocab_size = len(vocab["question_answer_token_to_idx"])
        self.linguistic_input_unit = TextualInputModule(
            vocab_size=encoder_vocab_size,
            wordvec_dim=word_dim,
            module_dim=module_dim,
            rnn_dim=module_dim,
        )
        self.visual_input_unit = StillVisualInputModule(
            k_max_frame_level=k_max_frame_level,
            k_max_clip_level=k_max_clip_level,
            spl_resolution=spl_resolution,
            vision_dim=vision_dim,
            module_dim=module_dim,
        )
        self.output_unit = OutputUnitMultiChoices(module_dim=module_dim)

        init_modules(self.modules(), w_init="xavier_uniform")
        nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)

    def forward(
        self,
        video_appearance_feat,
        video_motion_feat,
        question,
        question_len,
    ):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 4, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 4)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)

        question_embedding = self.linguistic_input_unit(question, question_len)
        visual_embedding = self.visual_input_unit(
            video_appearance_feat, video_motion_feat, question_embedding
        )

        visual_embedding = self.feature_aggregation(
            question_embedding, visual_embedding
        )

        out = self.output_unit(question_embedding, visual_embedding)

        return out
