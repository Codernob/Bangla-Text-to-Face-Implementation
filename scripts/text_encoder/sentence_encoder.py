#1. use this script for using bangla fasttext as text encoder (and comment scripts 2,3,4)

import torch
import numpy as np
import BanglaFastText


class SentenceEncoder:
    def __init__(self, device):
        self.Bn = BanglaFastText.BanglaFasttext(model_path = 'C:/Users/asus/sentence encoder/bangla fasttext/Bangla_FastText_skipgram.pickle')
        self.device = device

    def convert_text_to_embeddings(self, batch_text):
        stack = []
        #print('batch text:',batch_text)
        for sent in batch_text:
            #print('sent: ',sent)
            sentence_embeddings = self.Bn.sent_embd([sent])
            sentence_emb = torch.FloatTensor(sentence_embeddings).to(self.device).reshape(1,-1)
            stack.append(sentence_emb)
        output = torch.cat(stack, dim=0)
        return output.detach()


#2. use this script for using bangla fasttext with mean embedding as text encoder and comment out script 1,3,4
# import torch
# import numpy as np
# import BanglaFastText

# class SentenceEncoder:
#     def __init__(self, device):
#         self.Bn = BanglaFastText.BanglaFasttext(model_path = 'C:/Users/asus/sentence encoder/bangla fasttext/Bangla_FastText_skipgram.pickle')
#         self.device = device


#     def convert_text_to_embeddings(self, batch_text):
#         stack = []
#         #print('batch text:',batch_text)
#         for sent in batch_text:
#             l = sent.split("। ")
#             #print('sent: ',sent)
#             sentence_embeddings = self.Bn.sent_embd([l])
#             sentence_emb = torch.FloatTensor(sentence_embeddings).to(self.device)
#             sent_mean = torch.mean(sentence_emb,dim=0).reshape(1,-1)
#             stack.append(sent_mean)
#         output = torch.cat(stack, dim=0)
#         return output.detach()


# 3. use this script for using sbnltk sentence transformer hd as text encoder and comment out script 1, 2, 4
# import torch
# import numpy as np
# from sbnltk.Sentence_embedding import Bangla_sentence_embedding_hd

# class SentenceEncoder:
#     def __init__(self, device):
#         self.s2s = Bangla_sentence_embedding_hd()
#         self.device = device

#     def convert_text_to_embeddings(self, batch_text):
#         stack = []
#         #print('batch text:',batch_text)
#         for sent in batch_text:
#             #print('sent: ',sent)
#             sent = [sent]
#             sentence_embeddings = self.s2s.encode_sentence_list(sent)
#             ea = np.array([])
#             for key, value in sentence_embeddings.items():
#                 ea = np.hstack((ea,value))
#             sentence_emb = torch.FloatTensor(ea).to(self.device).reshape(1,-1)
#             stack.append(sentence_emb)
#         output = torch.cat(stack, dim=0)
#         return output.detach()

# 4. use this script for using sbnltk sentence transformer gd as text encoder and comment out script 1, 2, 3
# import torch
# import numpy as np
# from sbnltk.Sentence_embedding import Bangla_sentence_embedding_gd

# class SentenceEncoder:
#     def __init__(self, device):
#         self.s2s = Bangla_sentence_embedding_gd()
#         self.device = device

#     def convert_text_to_embeddings(self, batch_text):
#         stack = []
#         #print('batch text:',batch_text)
#         for sent in batch_text:
#             #print('sent: ',sent)
#             sent = [sent]
#             sentence_embeddings = self.s2s.encode_sentence_list(sent)
#             ea = np.array([])
#             for key, value in sentence_embeddings.items():
#                 ea = np.hstack((ea,value))
#             sentence_emb = torch.FloatTensor(ea).to(self.device).reshape(1,-1)
#             stack.append(sentence_emb)
#         output = torch.cat(stack, dim=0)
#         return output.detach()