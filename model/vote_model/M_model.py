import torch


class SharedMLPwEnsemble(torch.nn.Module):
    def __init__(self, args):
        super(SharedMLPwEnsemble, self).__init__()
        self.args = args
        self.base_dim = 768

        self.feature_projection = torch.nn.Linear(self.base_dim * 3, self.base_dim)  # 图像特征转换为GPT-2的嵌入维度

        self.video_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.video_dim, self.base_dim * 3),
            torch.nn.GELU()
        )

        self.audio_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.audio_dim, self.base_dim * 3),
            torch.nn.GELU()
        )

        self.text_adapter = torch.nn.Sequential(
            torch.nn.Linear(args.text_dim, self.base_dim * 3),
            torch.nn.GELU()
        )
        
        self.ensemble = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.base_dim * 3, self.base_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.base_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, args.target_dim)
            ) for _ in range(32)
        ])

    def forward(self, audio_feat, video_feat, text_feat):
        # Get batch size, sequence length and feature dimension
        B, T, _ = video_feat.shape
        video_feat = video_feat.reshape(B * T, -1)
        text_feat = text_feat.reshape(B * T, -1)
        audio_feat = audio_feat.reshape(B * T, -1)
        # Project features through adapters and feature projection
        video_feat = self.feature_projection(self.video_adapter(video_feat))  # [B*T, D]
        audio_feat = self.feature_projection(self.audio_adapter(audio_feat))  # [B*T, D]
        text_feat = self.feature_projection(self.text_adapter(text_feat))  # [B*T, D]
  
        multi_modal_chunk = torch.cat([video_feat, text_feat, audio_feat], dim=-1)  # [B*T, D*3]
        outputs = torch.stack([mlp(multi_modal_chunk) for mlp in self.ensemble], dim=0) # [32, B*T, target_dim]
        logits = outputs.mean(dim=0)  # [B*T, target_dim]
        logits = logits.view(B, T, -1).mean(dim=1)  # [B, target_dim]
        return logits
