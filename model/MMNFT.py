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
        glove_matrix,
        rnn_dim=512,
        out_dim=256,
        bidirectional=True,
    ):
        super(TextualInputModule, self).__init__()

        self.bidirectional = bidirectional
        if bidirectional:
            half_rnn_dim = rnn_dim // 2

        vocab_size, wordvec_dim = glove_matrix.size()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim).from_pretrained(
            glove_matrix, freeze=True
        )
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(
            wordvec_dim,
            half_rnn_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.text_dropout = nn.Dropout(p=0.18)
        self.fc = nn.Linear(rnn_dim, out_dim)

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

        # NOTE: pack_padded_sequence explanation
        # Ref: https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        embed = nn.utils.rnn.pack_padded_sequence(
            input=embed,  # padded batch of variable length sequences
            lengths=text_len.cpu().numpy(),  #  list of sequence lengths of each batch element
            batch_first=True,  # if True, the input is expected in B x T x * format.
            enforce_sorted=False,  # If False, the input will get sorted unconditionally
        )

        self.encoder.flatten_parameters()
        _, (embedding, _) = self.encoder(embed)

        if self.bidirectional:
            embedding = torch.cat([embedding[0], embedding[1]], -1)

        embedding = self.text_dropout(embedding)
        output = self.fc(embedding)

        # print(f"TextualInputModule output.shape: {output.shape}")
        return output


class StillVisualInputModule(nn.Module):
    def __init__(self, in_dim: int = 2048, out_dim: int = 256):
        super(StillVisualInputModule, self).__init__()

        self.dropout = nn.Dropout(p=0.15)
        self.fc = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.activation = nn.ReLU()

    def forward(self, feat):
        """
        Args:
            feat: [Tensor] (batch_size, in_dim)
        return:
            output representation [Tensor] (batch_size, out_dim)
        """
        feat = self.dropout(feat)
        output = self.activation(self.fc(feat))

        # print(f"StillVisualInputModule output.shape: {output.shape}")
        return output


class MotionVisualInputModule(nn.Module):
    def __init__(
        self,
        in_frames: int = 16,
        in_dim: int = 512,
        mid_dim: int = 32,
        out_dim: int = 256,
    ):
        super(MotionVisualInputModule, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=mid_dim, kernel_size=1)
        self.temporal = nn.Linear(in_features=in_frames * mid_dim, out_features=out_dim)

    def forward(self, feat):
        """
        Args:
            feat: [Tensor] (batch_size, in_frames, in_dim)
        return:
            output representation [Tensor] (batch_size, out_dim)
        """
        batch_size = feat.shape[0]
        feat = feat.permute(0, 2, 1)  # (batch_size, in_dim, in_frames)
        feat = self.conv(feat)  # (batch_size, mid_dim, in_frames)
        feat = feat.permute(0, 2, 1).reshape(
            batch_size, -1
        )  # (batch_size, in_frames*mid_dim)
        output = self.temporal(feat)  # (batch_size, out_dim)

        # print(f"MotionVisualInputModule output.shape: {output.shape}")
        return output


class AudioInputModule(nn.Module):
    def __init__(self, in_dim: int = 1000, out_dim: int = 256):
        super(AudioInputModule, self).__init__()
        # TODO: Implement Audio Input Module
        ...
        raise NotImplementedError()

    def forward(self, audio_feat):
        # TODO: Implement Audio Input Module
        ...
        raise NotImplementedError()


class FeatureAggregation(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super(FeatureAggregation, self).__init__()

        self.t_proj = nn.Linear(in_dim, mid_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, mid_dim, bias=False)
        self.m_proj = nn.Linear(in_dim, mid_dim, bias=False)

        self.fusion = nn.Linear(3 * mid_dim, out_dim)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, textual_feat, visual_feat, motion_feat):
        """
        Args:
            textual_feat: [Tensor] (batch_size, in_dim)
            visual_feat: [Tensor] (batch_size, in_dim)
            motion_feat: [Tensor] (batch_size, in_dim)
        return:
            output representation [Tensor] (batch_size, out_dim)
        """
        t_proj = self.t_proj(textual_feat)
        v_proj = self.v_proj(visual_feat)
        m_proj = self.m_proj(motion_feat)
        cat_feat = torch.cat([t_proj, v_proj, m_proj], dim=1)

        output = self.fusion(cat_feat)
        output = self.activation(self.dropout(output))

        # print(f"FeatureAggregation output.shape: {output.shape}")
        return output


class CLSOutputModule(nn.Module):
    """Classification Output Module"""

    def __init__(self, module_dim=512, num_classes=10):
        super(CLSOutputModule, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(module_dim, module_dim),
            nn.ELU(),
            nn.BatchNorm1d(module_dim),
            nn.Dropout(0.15),
            nn.Linear(module_dim, num_classes),
        )

    def forward(self, aggregated_feat):
        pred = self.classifier(aggregated_feat)
        return pred


class MMNFT(nn.Module):
    def __init__(
        self,
        glove_matrix: torch.Tensor,
        text_rnn_dim: int,
        visual_in_dim: int,
        motion_in_frames: int,
        motion_in_dim: int,
        motion_mid_dim: int,
        agg_in_dim: int,
        agg_mid_dim: int,
        agg_out_dim: int,
        num_classes: int,
        task: str = "classification",
    ):
        super(MMNFT, self).__init__()

        self.textual_input_module = TextualInputModule(
            glove_matrix=glove_matrix,
            rnn_dim=text_rnn_dim,
            out_dim=agg_in_dim,
        )
        self.still_visual_input_module = StillVisualInputModule(
            in_dim=visual_in_dim,
            out_dim=agg_in_dim,
        )
        self.motion_visual_input_module = MotionVisualInputModule(
            in_frames=motion_in_frames,
            in_dim=motion_in_dim,
            mid_dim=motion_mid_dim,
            out_dim=agg_out_dim,
        )
        self.feature_aggregation = FeatureAggregation(
            in_dim=agg_in_dim,
            mid_dim=agg_mid_dim,
            out_dim=agg_out_dim,
        )

        if task == "classification":
            self.output_module = CLSOutputModule(
                module_dim=agg_out_dim, num_classes=num_classes
            )
        else:
            raise NotImplementedError()

    def forward(self, text_encoded, text_length, image_feat, video_feat):
        # Feature Embedding
        textual_feat = self.textual_input_module(
            text_encoded, text_length
        )  # (B, agg_in_dim)
        image_feat = self.still_visual_input_module(image_feat)  # (B, agg_in_dim)
        video_feat = self.motion_visual_input_module(video_feat)  # (B, agg_in_dim)

        # Multi-modal Fusion
        agg_feat = self.feature_aggregation(textual_feat, image_feat, video_feat)

        # Classification / Regression Head
        out = self.output_module(agg_feat)

        return out
