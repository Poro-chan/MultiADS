import torch
from open_clip import tokenize

print(torch.cuda.is_available())

# Text prompts for learning (MVTec-AD)
# anomaly classes for MVTec: bent, color, crack, damaged, faulty imprint, hole, misplaced, poke, scratch, thread, broken, contamination, cut,
# fabric, glue, liquid, missing, rough, squeeze

# Mvtech templates
brisc2025_templates = {
    "glioma" : ['{} shows a glioma tumor', 'glioma found on {}', '{} with visible glioma', 'MRI shows a glioma in {}'],
    "meningioma" : ['{} with a meningioma tumor', 'meningioma found on {}', '{} shows a meningioma tumor', '{} with visible meningioma tumor'],
    "pituitary" : ['{} with a pituitary tumor', 'pituitary growth visible in {}', 'pituitary tumor detected in {}', '{} has a pituitary tumor']
}

mvtec_anomaly_templates = {
    "bent" : ['{} has a bent defect', 'flawed {} with a bent', 'a bend found in {}', '{} with noticeable bending'],
    "broken" : ['{} has a broken defect', 'flawed {} with breakage', 'visible breakage on {}', '{} with broken areas'],
    "color" : ['{} has a color defect', 'inconsistent color on {}', '{} with color discrepancies', '{} has a noticeable color difference'],
    "combined" : ['{} has a combined defect', 'multiple issues with {}', '{} with mixed defects', '{} showing multiple imperfections'],
    "contamination" : ['{} has a contamination defect', 'foreign particles on {}', '{} is contaminated', '{} contains contaminants'],
    "crack" : ['{} has a crack defect', 'a crack is present on {}', 'cracked area on {}', '{} with noticeable cracking'],
    "damaged" : ['{} has a damaged defect', 'flawed {} with damage', '{} with visible damage', 'damaged areas on {}'],
    "faulty_imprint" : ['{} has a faulty imprint defect', '{} has a print defect', 'incorrect printing on {}', 'misaligned print on {}'],
    "hole" : ['{} has a hole defect', 'a hole on {}', 'visible hole on {}', '{} with punctures'],
    "misplaced" : ['{} has a misplaced defect', 'flawed {} with misplacing', '{} shows misalignment', 'misplaced parts on {}'],
    "poke" : ['{} has a poke defect', '{} has a poke insulation defect', 'visible poke mark on {}', '{} has puncture marks'],
    "scratch" : ['{} has a scratch defect', 'flawed {} with a scratch', 'visible scratches on {}', '{} with surface scratches'],
    "thread" : ['{} has a thread defect', 'flawed {} with a thread', 'loose threads on {}', '{} has visible threads'],
    "cut" : ['{} has a cut defect', 'cut marks on {}', '{} with visible cuts', 'a cut detected on {}'],
    "fabric" : ['{} has a fabric defect', '{} has a fabric border defect', '{} has a fabric interior defect', 'fabric quality issues on {}'],
    "glue" : ['{} has a glue defect', '{} has a glue strip defect', 'excess glue on {}', '{} with uneven glue application'],
    "liquid" : ['{} has a liquid defect', 'flawed {} with liquid', '{} with oil', 'liquid marks on {}'],
    "missing" : ['{} has a missing defect', 'flawed {} with something missing', '{} has missing components', 'missing parts on {}'],
    "rough" : ['{} has a rough defect', 'rough texture on {}', 'uneven surface on {}', '{} is coarser than expected'],
    "squeeze" : ['{} has a squeeze defect', 'flawed {} with a squeeze', 'squeezed area on {}', '{} has compression marks']
}

visa_anomaly_templates = {
    "damage" : ['{} has a damaged defect', 'flawed {} with damage', '{} shows signs of damage', 'damage found on {}'],
    "scratch" : ['{} has a scratch defect', 'flawed {} with a scratch', 'scratches visible on {}', '{} has surface scratches'],
    "breakage" : ['{} with a breakage defect', 'broken {}', '{} with broken defect', '{} shows breakage'],
    "burnt": ['{} with a burnt defect', '{} shows burn marks', 'burnt areas on {}', '{} with signs of burning'],
    "weird_wick" : ['{} with a weird wick defect', '{} has an unusual wick', 'the wick on {} appears odd', '{} with a strangely shaped wick'],
    "stuck" : ['{} with a stuck defect', '{} stuck together', '{} has stuck parts', 'adhesive issue causing {} to stick'],
    "crack" : ['{} with a crack defect', '{} has a visible crack', 'cracked areas on {}', '{} with surface cracking'],
    "wrong_place" : ['{} with defect that something on wrong place', '{} has a misplaced defect', 'flawed {} with misplacing'],
    "partical" : ['{} with particles defect', '{} has foreign particles', 'small particles on {}', '{} with unwanted particles'],
    "bubble" : ['{} with bubbles defect', 'bubbles seen on {}', '{} with bubble marks', 'air bubbles in {}'],
    "melded" : ['{} with melded defect', 'melded parts on {}', '{} has fused areas', 'fused spots on {}'],
    "hole" : ['{} has a hole defect', 'a hole on {}', 'visible hole on {}', '{} has small punctures'],
    "melt" : ['{} with melt defect', 'melted areas on {}', '{} shows melting', 'signs of melting on {}'],
    "bent" : ['{} has a bent defect', 'flawed {} with a bent', 'bent areas on {}', '{} with visible bending'],
    "spot" : ['{} with spot defect', 'spots visible on {}', 'flawed {} with spots', '{} with visible spotting'],
    "extra" : ['{} with extra thing', '{} has a defect with extra thing', 'extra material on {}', '{} contains additional pieces'],
    "chip" : ['{} with chip defect', '{} with fragment broken defect', 'chipped areas on {}', '{} with chipped parts'],
    "missing" : ['{} with a missing defect', 'flawed {} with something missing', '{} has missing parts', 'missing components on {}']
}

class MultiADS_PromptLearner(torch.nn.Module) :
    def __init__(self, classnames, model, templates, details) :
        super().__init__()
        self.model = model
        device = next(model.parameters()).device
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.templates = templates
        self.n_ctx = details["Prompt_length"]
        n_ctx_pos = n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = details["learnabel_text_embedding_length"]
        self.compound_prompts_depth = details["learnabel_text_embedding_depth"]

        dtype = model.transformer.get_cast_dtype()
        ctx_dim = model.ln_final.weight.shape[0]

        self.state_normal_list = ['{}']
        self.normal_num = len(self.state_normal_list)
        self.ctx_pos = torch.nn.Parameter(torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype, device=device))
        torch.nn.init.normal_(self.ctx_pos, std=0.02)

        self.ctx_neg_dict = torch.nn.ParameterDict()
        for name, template in templates.items() :
            anomaly_num = len(template)
            self.ctx_neg_dict[name] = torch.nn.Parameter(torch.empty(self.n_cls, anomaly_num, n_ctx_neg, ctx_dim, dtype=dtype, device=device))
            torch.nn.init.normal_(self.ctx_neg_dict[name], std=0.02)
        
        self.compound_prompts_text = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim)) for _ in range (self.compound_prompts_depth - 1)])

        for p in self.compound_prompts_text :
            torch.nn.init.normal_(p, std=0.02)
        
        self.compound_prompt_projections = torch.nn.ModuleList(torch.nn.Linear(ctx_dim, 896) for i in range(self.compound_prompts_depth - 1))

        prompts_pos = []

        for name in self.classnames :
            for template in self.state_normal_list :
                prompts_pos.append(template.format(name) + ".")

        tokenized_prompts_pos = tokenize(prompts_pos).to(device)

        with torch.no_grad() :
            embedding_pos = model.token_embedding(tokenized_prompts_pos).type(dtype)
            n, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :].to(device) )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :].to(device) )

        self.tokenized_prompts_pos = tokenized_prompts_pos

        self.token_prefix_neg_dict = {}
        self.token_suffix_neg_dict = {}
        for name, template in templates.items() :
            prompts_neg = []
            for class_name in self.classnames :
                for text in template :
                    prompts_neg.append(text.format(class_name) + ".")
            tokenized_prompts_neg = tokenize(prompts_neg).to(device)
            with torch.no_grad() :
                embedding_neg = model.token_embedding(tokenized_prompts_neg).type(dtype)
                n, l, d = embedding_neg.shape
                embedding_neg = embedding_neg.reshape(self.n_cls, len(template), l, d)

            self.token_prefix_neg_dict[name] = embedding_neg[:, :, :1, :].to(device)
            self.token_suffix_neg_dict[name] = embedding_neg[:, :, 1 + n_ctx_neg:, :].to(device)

    
    def forward(self) :
        device = next(self.model.parameters()).device
        prefix_pos = self.token_prefix_pos.to(device)
        suffix_pos = self.token_suffix_pos.to(device)

        prompts_dict = {}
        for i, name in enumerate(self.classnames) :
            pos = torch.cat([prefix_pos[i:i+1], self.ctx_pos[i:i+1], suffix_pos[i:i+1]], dim=2)
            pos = pos.squeeze(0)
            pos_flat = pos.view(-1, pos.shape[-1])

            neg_list = []
            for anomaly_name in self.templates.keys() :
                prefix_neg = self.token_prefix_neg_dict[anomaly_name][i:i+1]
                suffix_neg = self.token_suffix_neg_dict[anomaly_name][i:i+1]
                ctx_neg = self.ctx_neg_dict[anomaly_name][i:i+1]
                neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=2).squeeze(0)
                neg_list.append(neg.view(-1, neg.shape[-1]))
            neg_flat = torch.cat(neg_list, dim=0)

            prompts_dict[name] = torch.cat([pos_flat, neg_flat], dim=0).T.to(self.ctx_pos.device)

        return prompts_dict