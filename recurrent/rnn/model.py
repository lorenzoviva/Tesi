import torch.nn as nn

from utils.embed_regularize import embedded_dropout
from utils.locked_dropout import LockedDropout

from dnc.dnc import DNC

class DNCModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
        ntoken,
        ninp,
        nhid,
        nlayers,
        nhlayers,
        dropout=0.5,
        dropouth=0.5,
        dropouti=0.5,
        dropoute=0.1,
        wdrop=0,
        tie_weights=False,
        nr_cells=5,
        read_heads=2,
        sparse_reads=10,
        cell_size=10,
        gpu_id=-1,
        rnn_type='rnn',
        controllers=None,
        independent_linears=False,
        debug=True
    ):
        super(DNCModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.debug = debug
        self.child = DNC(
                    input_size=ninp,
                    hidden_size=nhid,
                    num_layers=nlayers,
                    num_hidden_layers=nhlayers,
                    rnn_type=rnn_type,
                    controllers=controllers,
                    nr_cells=nr_cells,
                    read_heads=read_heads,
                    cell_size=cell_size,
                    gpu_id=gpu_id,
                    independent_linears=independent_linears,
                    debug=debug,
                    dropout=wdrop
                )

        print(self.child)

        self.decoder = nn.Linear(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, reset_experience=True):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        raw_output = emb
        # raw_output = input
        outputs = []
        if self.debug:
            debug_mems = []
        raw_output = raw_output.transpose(0, 1)
        if self.debug:
            raw_output, new_h, debug = self.child(raw_output, hidden[0], reset_experience=reset_experience, pass_through_memory=True)
            debug_mems.append(debug)
        else:
            raw_output, new_h = self.child(raw_output, hidden[0], reset_experience=reset_experience)
        raw_output = raw_output.transpose(0, 1)

        raw_output = self.lockdrop(raw_output, self.dropouth)
        outputs.append(raw_output)

        hidden = [new_h]

        output = self.lockdrop(raw_output, self.dropout).contiguous()
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            if self.debug:
                return result, hidden, raw_output, outputs, debug_mems
            return result, hidden, raw_output, outputs
        if self.debug:
            return result, hidden, debug_mems
        return result, hidden

