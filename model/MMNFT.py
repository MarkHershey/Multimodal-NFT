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
    def __init__(self, in_dim=1000, out_dim=500):
        super(StillVisualInputModule, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, image_feat):
        """
        Args:
            image_feat: [Tensor] (batch_size, in_dim)
        return:
            image_feat representation [Tensor] (batch_size, out_dim)
        """
        image_feat = self.fc(image_feat)
        image_feat = self.activation(image_feat)

        return image_feat


class MotionVisualInputModule(nn.Module):
    def __init__(self, in_dim=1000, out_dim=500):
        super(MotionVisualInputModule, self).__init__()

        self.temporal = nn.LSTM(
            input_size=in_dim,
            hidden_size=out_dim,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, motion_feat):

        output, (h_n, c_n) = self.temporal(motion_feat)
        print(f"MotionVisualInputModule temporal LSTM output.shape: {output.shape}")
        return output[-1]


class AudioInputModule(nn.Module):
    def __init__(self, dim):
        super(AudioInputModule, self).__init__()
        # TODO: Implement Audio Input Module
        ...
        raise NotImplementedError()

    def forward(self, audio_feat):
        # TODO: Implement Audio Input Module
        ...
        raise NotImplementedError()


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


class CLSOutputModule(nn.Module):
    """Classification Output Module"""

    def __init__(self, module_dim=512):
        super(CLSOutputModule, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(module_dim * 4, module_dim),
            nn.ELU(),
            nn.BatchNorm1d(module_dim),
            nn.Dropout(0.15),
            nn.Linear(module_dim, 1),
        )

    def forward(self, aggregated_feat):
        pred = self.classifier(aggregated_feat)


class MMNFT(nn.Module):
    def __init__(
        self,
        vocab_size,
        glove_matrix,
        wordvec_dim,
        module_dim,
        word_dim,
        visual_in_dim: int,
        motion_in_frames: int,
        motion_in_dim: int,
        vocab,
        task: str = "classification",
    ):
        super(MMNFT, self).__init__()

        self.textual_input_module = TextualInputModule(
            vocab_size=vocab_size,
            glove_matrix=glove_matrix,
            wordvec_dim=wordvec_dim,
        )
        self.still_visual_input_module = StillVisualInputModule()
        self.motion_visual_input_module = MotionVisualInputModule()
        self.feature_aggregation = FeatureAggregation(module_dim)

        if task == "classification":
            self.output_module = CLSOutputModule(module_dim)
        else:
            raise NotImplementedError()

        self.output_module = CLSOutputModule(module_dim=module_dim)

    def forward(
        self,
        video_appearance_feat,
        video_motion_feat,
        question,
        question_len,
    ):

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
