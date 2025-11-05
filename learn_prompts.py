import torch
from open_clip import tokenize

print(torch.cuda.is_available())

# Text prompts for learning (MVTec-AD)
# anomaly classes for MVTec: bent, color, crack, damaged, faulty imprint, hole, misplaced, poke, scratch, thread, broken, contamination, cut,
# fabric, glue, liquid, missing, rough, squeeze

mvtech_anomaly_templates = [
    'bent {}',
    'discolored {}',
    'cracked {}',
    'damaged {}',
    'faulty imprinted {}',
    'holed {}',
    'misplaced {}',
    'poked {}',
    'scratched {}',
    'threaded {}',
    'broken {}',
    'contaminated {}',
    'cut {}',
    'defective fabric {}',
    'glued {}',
    'liquid stained {}',
    'missing part {}',
    'rough {}',
    'squeezed {}'
]

visa_anomaly_templates = [
    'damaged {}',
    'scratched {}',
    'broken {}',
    'burnt {}',
    'odd wicked {}',
    'stuck {}',
    'cracked {}',
    'misplaced {}',
    'foreign particles {}',
    'bubbled {}',
    'melded {}',
    'holed {}',
    'melted {}',
    'bent {}',
    'spotted {}',
    'extraneous {}',
    'chipped {}',
    'missing part {}'
]

class MultiADS_PromptLearner(torch.nn.Module) :
    def __init__(self, classnames, model, templates, details) :
        super().__init__()
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = details["Prompt_length"]
        n_ctx_pos = n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = details["learnabel_text_embedding_length"]
        self.compound_prompts_depth = details["learnabel_text_embedding_depth"]

        dtype = model.transformer.get_cast_dtype()
        ctx_dim = model.ln_final.weight.shape[0]

        self.state_normal_list = ['{}']
        self.state_anomaly_list = templates

        self.normal_num = len(self.state_normal_list)
        self.anomaly_num = len(self.state_anomaly_list)

        self.ctx_pos = torch.nn.Parameter(torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype))
        self.ctx_neg = torch.nn.Parameter(torch.empty(self.n_cls, self.anomaly_num, n_ctx_pos, ctx_dim, dtype=dtype))
        torch.nn.init.normal_(self.ctx_pos, std=0.02)
        torch.nn.init.normal_(self.ctx_neg, std=0.02)

        self.compound_prompts_text = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim)) for _ in range (self.compound_prompts_depth - 1)])

        for p in self.compound_prompts_text :
            torch.nn.init.normal_(p, std=0.02)
        
        self.compound_prompt_projections = torch.nn.ModuleList(torch.nn.Linear(ctx_dim, 896) for i in range(self.compound_prompts_depth - 1))

        prompts_pos = []
        prompts_neg = []

        for name in self.classnames :
            for template in self.state_normal_list :
                prompts_pos.append(template.format(name) + ".")
            for template in self.state_anomaly_list :
                prompts_neg.append(template.format(name) + ".")

        tokenized_prompts_pos = tokenize(prompts_pos)
        tokenized_prompts_neg = tokenize(prompts_neg)

        with torch.no_grad() :
            embedding_pos = model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.anomaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        self.tokenized_prompts_pos = tokenized_prompts_pos
        self.tokenized_prompts_neg = tokenized_prompts_neg
    
    def forward(self) :
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        prompts_dict = {}
        for i, name in enumerate(self.classnames) :
            pos = torch.cat([prefix_pos[i:i+1], self.ctx_pos[i:i+1], suffix_pos[i:i+1]], dim=2)
            neg = torch.cat([prefix_neg[i:i+1], self.ctx_neg[i:i+1], suffix_neg[i:i+1]], dim=2)

            pos = pos.squeeze(0)
            neg = neg.squeeze(0)

            pos_flat = pos.view(-1, pos.shape[-1])
            neg_flat = neg.view(-1, neg.shape[-1])

            prompts = torch.cat([pos_flat, neg_flat], dim=0)
            prompts_dict[name] = prompts.T.to(self.ctx_pos.device)

        return prompts_dict