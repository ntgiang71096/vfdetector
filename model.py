from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

class PatchClassifier(nn.Module):
    def __init__(self):
        super(PatchClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.before_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.after_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.combine = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)

        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch):
        d1, d2, d3 = before_batch.shape
        before_batch = torch.reshape(before_batch, (d1, d2*d3))
        after_batch = torch.reshape(after_batch, (d1, d2*d3))

        before = self.before_linear(before_batch)
        after = self.after_linear(after_batch)
        combined = self.combine(torch.cat([before, after], axis=1))

        out = self.out_proj(combined)

        return out


class CnnClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[2, 3, 4],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(CnnClassifier, self).__init__()
        self.embed_dim = embed_dim
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(2 * np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, before_batch, after_batch):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed_before = before_batch
        # batch, file, hidden_dim

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_before = x_embed_before.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_before = [F.relu(conv1d(x_reshaped_before)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_before = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list_before]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_before = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_before],
                         dim=1)

        # # Compute logits. Output shape: (b, n_classes)
        # out = self.fc(self.dropout(x_fc_before))


        ############################################


        x_embed_after = after_batch

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_after = x_embed_after.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_after = [F.relu(conv1d(x_reshaped_after)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_after = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list_after]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_after = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_after],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)

        x_fc = torch.cat([x_fc_before, x_fc_after], axis=1)
        out = self.fc(self.dropout(x_fc))

        return out


class VariantTwoClassifier(nn.Module):
    def __init__(self):
        super(VariantTwoClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3

        self.linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, file_batch, need_final_feature=False):
        d1, d2, d3 = file_batch.shape
        file_batch = torch.reshape(file_batch, (d1, d2*d3))

        commit_embedding = self.linear(file_batch)

        x = commit_embedding
        x = self.relu(x)
        final_feature = x

        x = self.drop_out(x)
        out = self.out_proj(x)

        if need_final_feature:
            return out, final_feature
        else:
            return out

class VulFixMinerFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VulFixMinerFineTuneClassifier, self).__init__()
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.linear = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch).last_hidden_state[:, 0, :]
        x = self.linear(embeddings)
        x = self.drop_out(x)
        out = self.out_proj(x)
   
        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VulFixMinerClassifier(nn.Module):
    def __init__(self):
        super(VulFixMinerClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.linear = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, embedding_batch):
        x = self.linear(embedding_batch)
        x = self.drop_out(x)
        out = self.out_proj(x)
   
        return out

class VariantTwoFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantTwoFineTuneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantTwoClassifier()

    def forward(self, input_list_batch, mask_list_batch):
        d1, d2, d3 = input_list_batch.shape
        input_list_batch = torch.reshape(input_list_batch, (d1 * d2, d3))
        mask_list_batch = torch.reshape(mask_list_batch, (d1 * d2, d3))
        embeddings = self.code_bert(input_ids=input_list_batch, attention_mask=mask_list_batch).last_hidden_state[:, 0, :]
        embeddings = torch.reshape(embeddings, (d1, d2, self.HIDDEN_DIM))

        out = self.classifier(embeddings)

        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VariantSixClassifier(nn.Module):
    def __init__(self):
        super(VariantSixClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3

        self.before_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.after_linear = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.combine = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)

        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.HIDDEN_DIM, 2)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        d1, d2, d3 = before_batch.shape
        before_batch = torch.reshape(before_batch, (d1, d2*d3))
        after_batch = torch.reshape(after_batch, (d1, d2*d3))

        before = self.before_linear(before_batch)
        after = self.after_linear(after_batch)
        combined = self.combine(torch.cat([before, after], axis=1))

        x = combined
        x = self.relu(x)

        final_feature = x

        x = self.drop_out(x)
        out = self.out_proj(x)

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantThreeClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(VariantThreeClassifier, self).__init__()
        self.embed_dim = embed_dim
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, code, need_final_feature=False):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = code

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        final_feature = x_fc

        out = self.fc(self.dropout(x_fc))

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantThreeFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantThreeFineTuneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantThreeClassifier()

    def forward(self, input_list_batch, mask_list_batch):
        d1, d2, d3 = input_list_batch.shape
        input_list_batch = torch.reshape(input_list_batch, (d1 * d2, d3))
        mask_list_batch = torch.reshape(mask_list_batch, (d1 * d2, d3))
        embeddings = self.code_bert(input_ids=input_list_batch, attention_mask=mask_list_batch).last_hidden_state[:, 0, :]
        embeddings = torch.reshape(embeddings, (d1, d2, self.HIDDEN_DIM))

        out = self.classifier(embeddings)

        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VariantThreeFineTuneOnlyClassifier(nn.Module):
    def __init__(self):
        super(VariantThreeFineTuneOnlyClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)

        self.linear = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch).last_hidden_state[:, 0, :]

        x = embeddings
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x

class VariantSevenClassifier(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 embed_dim=768,
                 filter_sizes=[2, 3, 4],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        super(VariantSevenClassifier, self).__init__()
        self.embed_dim = embed_dim
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(2 * np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed_before = before_batch
        # batch, file, hidden_dim

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_before = x_embed_before.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_before = [F.relu(conv1d(x_reshaped_before)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_before = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list_before]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_before = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_before],
                         dim=1)

        # # Compute logits. Output shape: (b, n_classes)
        # out = self.fc(self.dropout(x_fc_before))


        ############################################


        x_embed_after = after_batch

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped_after = x_embed_after.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list_after = [F.relu(conv1d(x_reshaped_after)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list_after = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list_after]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc_after = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list_after],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)

        x_fc = torch.cat([x_fc_before, x_fc_after], axis=1)
        final_feature = x_fc

        out = self.fc(self.dropout(x_fc))

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantOneClassifier(nn.Module):
    def __init__(self):
        super(VariantOneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2
        self.linear = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, embedding_batch, need_final_feature=False):
        x = embedding_batch
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        final_feature = x
        x = self.drop_out(x)
        x = self.out_proj(x)

        if need_final_feature:
            return x, final_feature
        else:
            return x

class VariantOneFinetuneClassifier(nn.Module):
    def __init__(self):
        super(VariantOneFinetuneClassifier, self).__init__()

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantOneClassifier()

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch)
        embeddings = embeddings.last_hidden_state[:, 0, :]
        out = self.classifier(embeddings)
        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VariantFiveClassifier(nn.Module):
    def __init__(self):
        super(VariantFiveClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.5
        self.NUMBER_OF_LABELS = 2
        self.linear = nn.Linear(2 * self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        combined = torch.cat([before_batch, after_batch], dim=1)
        x = combined
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        final_feature = x
        x = self.drop_out(x)
        x = self.out_proj(x)

        if need_final_feature:
            return x, final_feature
        else:
            return x


class VariantFiveFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantFiveFineTuneClassifier, self).__init__()
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantFiveClassifier()

    def forward(self, added_input, added_mask, removed_input, removed_mask):
        added_embeddings = self.code_bert(input_ids=added_input, attention_mask=added_mask).last_hidden_state[:, 0, :]
        removed_embeddings = self.code_bert(input_ids=removed_input, attention_mask=removed_mask).last_hidden_state[:, 0, :]
        out = self.classifier(added_embeddings, removed_embeddings)
        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class VariantEightClassifier(nn.Module):
    def __init__(self):
        super(VariantEightClassifier, self).__init__()
        self.input_size = 768
        self.hidden_size = 128
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(4 * self.hidden_size, self.hidden_size)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)

        self.out_proj = nn.Linear(self.hidden_size, 2)

    def forward(self, before_batch, after_batch, need_final_feature=False):
        # self.lstm.flatten_parameters()
        before_out, (before_final_hidden_state, _) = self.lstm(before_batch)
        before_vector = before_out[:, 0]

        after_out, (after_final_hidden_state, _) = self.lstm(after_batch)
        after_vector = after_out[:, 0]

        x = self.linear(torch.cat([before_vector, after_vector], axis=1))

        x = self.relu(x)
        final_feature = x

        x = self.drop_out(x)

        out = self.out_proj(x)

        if need_final_feature:
            return out, final_feature
        else:
            return out


class VariantSixFineTuneClassifier(nn.Module):
    def __init__(self):
        super(VariantSixFineTuneClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)
        self.classifier = VariantSixClassifier()

    def forward(self, added_input_list_batch, added_mask_list_batch, removed_input_list_batch, removed_mask_list_batch):
        d1, d2, d3 = added_input_list_batch.shape
        added_input_list_batch = torch.reshape(added_input_list_batch, (d1 * d2, d3))
        added_mask_list_batch = torch.reshape(added_mask_list_batch, (d1 * d2, d3))
        added_embeddings = self.code_bert(input_ids=added_input_list_batch, attention_mask=added_mask_list_batch).last_hidden_state[:, 0, :]
        added_embeddings = torch.reshape(added_embeddings, (d1, d2, self.HIDDEN_DIM))

        removed_input_list_batch = torch.reshape(removed_input_list_batch, (d1 * d2, d3))
        removed_mask_list_batch = torch.reshape(removed_mask_list_batch, (d1 * d2, d3))
        removed_embeddings = self.code_bert(input_ids=removed_input_list_batch, attention_mask=removed_mask_list_batch).last_hidden_state[:, 0, :]
        removed_embeddings = torch.reshape(removed_embeddings, (d1, d2, self.HIDDEN_DIM))

        out = self.classifier(added_embeddings, removed_embeddings)

        return out

    def freeze_codebert(self):
        if not isinstance(self, nn.DataParallel):
            for param in self.code_bert.parameters():
                param.requires_grad = False
        else:
            for param in self.module.code_bert.parameters():
                param.requires_grad = False


class EncoderRNN(nn.Module):
    def __init__(self,
                 emb_dim,
                 h_dim,
                 batch_first=True):
        super(EncoderRNN, self).__init__()
        self.h_dim = h_dim
        # self.embed = nn.Embedding(v_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim,
                            h_dim,
                            batch_first=batch_first,
                            bidirectional=True)
    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        c0 = Variable(torch.zeros(1*2, b_size, self.h_dim))
        h0 = h0.cuda()
        c0 = c0.cuda()
        return h0, c0

    def forward(self, sentence):
        # self.lstm.flatten_parameters()
        hidden = self.init_hidden(sentence.size(0))
        emb = sentence
        out, hidden = self.lstm(emb, hidden)
        out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        return out


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(nn.Linear(h_dim, 24), nn.ReLU(True),
                                  nn.Linear(24, 1))

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.reshape(-1, self.h_dim)) # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.view(b_size, -1),
                         dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)


class AttnClassifier(nn.Module):
    def __init__(self, h_dim, c_num):
        super(AttnClassifier, self).__init__()
        self.attn1 = Attn(h_dim)
        self.attn2 = Attn(h_dim)
        self.linear = nn.Linear(2*h_dim, h_dim)
        self.output = nn.Linear(h_dim, c_num)

    def forward(self, a_output, b_output):
        a_attn = self.attn1(a_output)  #(b, s, 1)
        b_attn = self.attn2(b_output) #()  #(b, s, 1)
        a_feats = (a_output * a_attn).sum(dim=1)  # (b, s, h) -> (b, h)
        b_feats = (b_output * b_attn).sum(dim=1)
        feats = torch.cat((a_feats, b_feats), 1)
        o_feats = self.linear(feats)
        out = self.output(o_feats)
        return F.log_softmax(out, -1) , a_attn, b_attn


class VariantEightAttentionClassifier(nn.Module):
    def __init__(self):
        super(VariantEightAttentionClassifier, self).__init__()
        self.EMBEDDING_DIM = 768
        self.HIDDEN_DIM = 128
        self.NUMBER_OF_LABELS = 2
        self.before_encoder = EncoderRNN(self.EMBEDDING_DIM, self.HIDDEN_DIM)
        self.after_encoder = EncoderRNN(self.EMBEDDING_DIM, self.HIDDEN_DIM)

        self.attn1 = Attn(self.HIDDEN_DIM)
        self.attn2 = Attn(self.HIDDEN_DIM)
        self.linear = nn.Linear(2 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.output = nn.Linear(self.HIDDEN_DIM, self.NUMBER_OF_LABELS)

    def forward(self, before_batch, after_batch):
        before_out = self.before_encoder(before_batch)
        after_out = self.after_encoder(after_batch)
        a_attn = self.attn1(after_out)  # (b, s, 1)
        b_attn = self.attn2(before_out)  # ()  #(b, s, 1)
        a_feats = (after_out * a_attn).sum(dim=1)  # (b, s, h) -> (b, h)
        b_feats = (before_out * b_attn).sum(dim=1)
        feats = torch.cat((a_feats, b_feats), 1)
        o_feats = self.linear(feats)
        out = self.output(o_feats)
        return out

class VariantEightFineTuneOnlyClassifier(nn.Module):
    def __init__(self):
        super(VariantEightFineTuneOnlyClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)

        self.linear = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch).last_hidden_state[:, 0, :]

        x = embeddings
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x


class VariantSeventFineTuneOnlyClassifier(nn.Module):
    def __init__(self):
        super(VariantSeventFineTuneOnlyClassifier, self).__init__()
        self.HIDDEN_DIM = 768
        self.DENSE_DIM = 768
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2

        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)

        self.linear = nn.Linear(self.HIDDEN_DIM, self.DENSE_DIM)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.DENSE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, input_batch, mask_batch, need_final_feature=False):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch).last_hidden_state[:, 0, :]

        x = embeddings
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.relu(x)
        final_feature = x
        x = self.drop_out(x)
        x = self.out_proj(x)

        if need_final_feature:
            return x, final_feature
        else:
            return x


class EnsembleModel(nn.Module):
    def __init__(self, ablation_study=False, variant_to_drop=None):
        super(EnsembleModel, self).__init__()
        self.FEATURE_DIM = 768
        self.DENSE_DIM = 128
        self.CNN_FEATURE_DIM = 300
        self.HIDDEN_DIM_DROPOUT_PROB = 0.3
        self.NUMBER_OF_LABELS = 2
        # need 2 linear layer to project CNN feature dim to 768
        # 1 for variant 3
        # 1 for variant 7
        self.l1 = nn.Linear(self.CNN_FEATURE_DIM, self.FEATURE_DIM)
        self.l2 = nn.Linear(self.CNN_FEATURE_DIM * 2, self.FEATURE_DIM)

        # need 1 linear layer to project variant 5 feature to 768

        self.l3 = nn.Linear(self.DENSE_DIM, self.FEATURE_DIM)

        # need 1 linear layer to project variant 8 feature to 768
        self.l4 = nn.Linear(self.DENSE_DIM, self.FEATURE_DIM)

        # 1 layer to combine
        self.ablation_study = ablation_study

        if not self.ablation_study:
            self.l5 = nn.Linear(7 * self.FEATURE_DIM, self.FEATURE_DIM)
        else:
            self.l5 = nn.Linear((7 - len(variant_to_drop)) * self.FEATURE_DIM, self.FEATURE_DIM)

        self.variant_to_drop = variant_to_drop

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(self.HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(self.FEATURE_DIM, self.NUMBER_OF_LABELS)

    def forward(self, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8):
        feature_3 = self.l1(feature_3)
        feature_7 = self.l2(feature_7)
        feature_5 = self.l3(feature_5)
        feature_8 = self.l4(feature_8)
        all_features = [feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8]
        if self.ablation_study:
            tmp = all_features
            all_features = []
            drop = []
            drop.append(True) if 1 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 2 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 3 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 5 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 6 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 7 in self.variant_to_drop else drop.append(False)
            drop.append(True) if 8 in self.variant_to_drop else drop.append(False)
            for i in range(len(drop)):
                if not drop[i]:
                    all_features.append(tmp[i])
        feature_list = torch.cat(all_features, axis=1)
        x = self.drop_out(feature_list)
        x = self.l5(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x