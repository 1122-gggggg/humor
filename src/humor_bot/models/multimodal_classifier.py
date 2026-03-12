"""
StandUp4AI 多模態幽默檢測模型

基於論文的 Sequence Classification 架構，實作多模態特徵融合：
1. 視覺特徵 (Vision, V): 演員死魚臉/誇張動作
2. 音訊特徵 (Audio, A): 停頓、重音
1. 二元結構預測: f(C, P) = y
   將前 n-1 句視為 Context (C)，最後一句視為 Punchline (P)，預測是否好笑。
2. 幽默觸發點時序預測: f(T_<=t, A_<=t, V_<=t) = y_t
3. 結構與技術序列標註: P(Y|X) = exp(sum(λ * f(y_i, y_{i-1}, X))) / Z(X)
   使用 Transformer-Encoder 配合線性映射與可學習轉移矩陣 (CRF-like features) 
   進行 Setup -> Punchline 等上下文的動態推演。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    跨模態注意力機制 (Cross-Modal Attention)
    
    論文精華：模型在聽(Q)的時候，會去另一模態(K, V)中尋找對應的線索。
    例如使用 Text 增強 Audio 特徵：
    Attention(Q_A, K_T, V_T) = softmax(Q_A * K_T^T / sqrt(d_k)) * V_T
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
            )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: 查詢張量，例如 Audio 特徵 (batch, seq_len, d_model)
            key: 鍵張量，例如 Text 特徵 (batch, seq_len, d_model)
            value: 值張量，例如 Text 特徵 (batch, seq_len, d_model)
        Returns:
            enhanced_feature: 增強後的特徵 (batch, seq_len, d_model)
        """
        attn_out, _ = self.mha(query, key, value)
        return attn_out


class MultimodalIncongruityModule(nn.Module):
    """
    模態衝突偵測模組 (Multimodal Incongruity for Sarcasm)
    
    諷刺的本質在於語義 (Text) 與表達 (Audio, Visual) 的強烈不協調。
    例如 Text (+) 極度正面，但 Audio (-) 與 Visual (-) 非常負面或冷漠。
    透過計算 Text 特徵與 Audio/Visual 特徵之間的歐氏距離差距與相互投影，
    來明確量化出這股「不協調 (Incongruity)」的數學變量。
    """
    def __init__(self, d_model: int):
        super().__init__()
        # 包含 (T-A), (T-V), (T*A), (T*V) 四種交互，故為 d_model * 4
        self.conflict_proj = nn.Linear(d_model * 4, d_model)
        
    def forward(self, T_h: torch.Tensor, A_h: torch.Tensor, V_h: torch.Tensor) -> torch.Tensor:
        # 1. 歐氏距離差距 (絕對差值)
        text_audio_diff = torch.abs(T_h - A_h)
        text_vision_diff = torch.abs(T_h - V_h)
        
        # 2. 元素級交互 (Element-wise product, MUStARD 常用技巧)
        text_audio_mul = T_h * A_h
        text_vision_mul = T_h * V_h
        
        # 拼接並提取衝突/協同特徵
        conflict_concat = torch.cat([text_audio_diff, text_vision_diff, text_audio_mul, text_vision_mul], dim=-1)
        incongruity_feature = self.conflict_proj(conflict_concat) # (batch, seq_len, d_model)
        
        return incongruity_feature


class TensorFusionNetwork(nn.Module):
    """
    多模態張量融合網路 (Tensor Fusion Network, TFN)
    
    理論依據：MUStARD / UR-FUNNY 論文
    實作公式：Z = [T_h, 1] ⊗ [A_h, 1] ⊗ [V_h, 1]
    
    物理意義：
    - 單模態項 (Unimodal): 捕捉單純文字好笑、或單純表情好笑。
    - 雙模態交互 (Bimodal): 捕捉「文字+語氣」的衝突或協同。
    - 三模態交互 (Trimodal): 捕捉最高階的幽默，即「文字的冷、語氣的靜、表情的崩潰」三位一體。
    """
    def __init__(self, d_model: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        # TFN 產生的張量維度為 (d_model + 1)^3，這通常非常大 (如 257^3 > 16M)
        # 在工程實務上，為避免 OOM，我們使用降維或適度壓縮的特徵來模擬 TFN
        # 這裡為了展示張量外積精髓且兼顧效率，我們在進行外積前先將模態維度降至 `reduced_dim`
        # （否則一般會使用 Low-rank Tensor Fusion, LMF）
        self.reduced_dim = 16 
        
        self.compress_t = nn.Linear(d_model, self.reduced_dim)
        self.compress_a = nn.Linear(d_model, self.reduced_dim)
        self.compress_v = nn.Linear(d_model, self.reduced_dim)
        
        # 張量展開後的維度
        tensor_dim = (self.reduced_dim + 1) * (self.reduced_dim + 1) * (self.reduced_dim + 1)
        
        self.post_fusion_dropout = nn.Dropout(dropout)
        self.project_to_output = nn.Linear(tensor_dim, output_dim)

    def forward(self, T_h: torch.Tensor, A_h: torch.Tensor, V_h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T_h, A_h, V_h: (batch_size, seq_len, d_model)
        Returns:
            fused_out: (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = T_h.shape
        
        # 1. 降維以利外積展開 (B, L, reduced_dim)
        t_comp = self.compress_t(T_h)
        a_comp = self.compress_a(A_h)
        v_comp = self.compress_v(V_h)
        
        # 2. 為每個模態增加一個恆為 1 的維度，以保留 Unimodal 與 Bimodal 特徵
        # [ones, features] -> (B, L, reduced_dim + 1)
        ones = torch.ones(batch_size, seq_len, 1, device=T_h.device, dtype=T_h.dtype)
        
        t_ext = torch.cat([ones, t_comp], dim=-1) # (B, L, 17)
        a_ext = torch.cat([ones, a_comp], dim=-1) # (B, L, 17)
        v_ext = torch.cat([ones, v_comp], dim=-1) # (B, L, 17)
        
        # 3. 三階張量外積 (Kronecker Product / Outer Product)
        # 我們利用 bmm 或直接使用 einsum 來計算 batch_size x seq_len 下的三階外積
        # Einsum 公式： b l i, b l j, b l k -> b l (i j k)
        # 這裡分兩步做：先 (T ⊗ A)，再和 V 做 ⊗
        
        # Step 3.1: t_ext ⊗ a_ext
        # (B, L, i, 1) @ (B, L, 1, j) -> (B, L, i, j)
        fusion_ta = torch.matmul(t_ext.unsqueeze(-1), a_ext.unsqueeze(2)) 
        fusion_ta_flat = fusion_ta.view(batch_size, seq_len, -1) # (B, L, 17*17)
        
        # Step 3.2: (T ⊗ A) ⊗ v_ext
        # (B, L, i*j, 1) @ (B, L, 1, k) -> (B, L, i*j, k)
        fusion_tensor = torch.matmul(fusion_ta_flat.unsqueeze(-1), v_ext.unsqueeze(2))
        
        # 4. 展平三階張量 (Flatten)
        fusion_flat = fusion_tensor.view(batch_size, seq_len, -1) # (B, L, 17*17*17 = 4913)
        
        # 5. 投影回模型維度 (Project back to d_model)
        fused_out = self.post_fusion_dropout(fusion_flat)
        fused_out = self.project_to_output(fused_out) # (B, L, output_dim)
        
        return fused_out


class FeatureGatingNetwork(nn.Module):
    """
    動態特徵門控網路 (MoE Router / Dynamic Gating)
    
    學術依據：基於最新的 Multimodal Mixture of Experts (MMoE, 2024) 架構。
    物理意義：
    每一種幽默類型需要的資訊來源不同：
    - 肢體搞笑 (Physical): 依賴 TFN 中的視覺與聲音張量交互。
    - 諷刺 (Satire/Sarcasm): 極度依賴 Incongruity 衝突變量。
    - 地獄梗/自嘲 (Dark/Self-Deprecation): 非常仰賴講者的 Persona 人設。
    此 Router 會根據當下的上下文，動態給予這三名 Expert 不同的注意力權重。
    """
    def __init__(self, d_model: int, num_experts: int = 3):
        super().__init__()
        # 輸入形狀：拼接後的所有 Expert 特徵
        self.router = nn.Sequential(
            nn.Linear(d_model * num_experts, d_model),
            nn.SiLU(), # Swish 激活函數在 MoE 路由表現通常較 ReLU 好
            nn.Linear(d_model, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, experts: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            experts: [TFN (B, L, D), Incongruity (B, L, D), Persona (B, L, D)]
        Returns:
            gated_concat_features (B, L, 3D)
        """
        concat_features = torch.cat(experts, dim=-1) # (B, L, 3D)
        
        # 計算每個 Expert 的動態權重 (B, L, num_experts)
        gating_weights = self.router(concat_features)
        
        weighted_experts = []
        for i, expert in enumerate(experts):
            # 擷取對應 Expert 的權重 (B, L, 1) 並乘上特徵
            # 乘以 len(experts) 是為了維持梯度與數值的 scale 穩定性
            weight = gating_weights[:, :, i:i+1] * len(experts)
            weighted_experts.append(expert * weight)
            
        return torch.cat(weighted_experts, dim=-1) # (B, L, 3D)


class MultimodalHumorClassifier(nn.Module):
    """
    多模態幽默序列分類器暨標註機 (Sequence Classification & Labeling)
    
    將幽默檢測視為時間序列分類，結合 T, A, V 預測 t 時刻是否觸發笑聲，
    並同時進行結構標籤 (Structural) 與技術標籤 (Technique) 的時序解析。
    加入了外部知識映射：Persona Embedding (講者人設)。
    """
    # 結構標籤 (Structural Tags)
    STRUCT_TAGS = {"O": 0, "ST": 1, "BR": 2, "PL": 3, "TL": 4}  # None, Setup, Bridge, Punchline, Tagline
    
    # 技術標籤 (Technique Tags)
    TECH_TAGS = {"O": 0, "Analogy": 1, "Persona": 2, "StatusSwitch": 3}
    def __init__(
        self, 
        text_dim: int = 768,     # e.g., RoBERTa/BERT hidden dim
        audio_dim: int = 128,    # e.g., VGGish/YAMNet feature dim
        vision_dim: int = 512,   # e.g., ResNet/OpenFace feature dim
        persona_dim: int = 256,  # 講者人設 Context 向量維度
        d_model: int = 256, 
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1. 各模態專屬編碼器 (投影到相同維度)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        
        # 外部知識：講者人設映射
        self.persona_proj = nn.Linear(persona_dim, d_model)
        
        # 2. 模態間的互相增強 (Cross-modal attention)
        # 論文指出：音頻(停頓/重音)通常比文本更能預測笑聲，視覺有輔助作用
        # 這邊實作 Text 對 Audio 的增強：以 Audio 為 Query，去 Text 尋找對應上下文
        self.text_to_audio_attn = CrossModalAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        
        # 實作 Vision 對 Audio 的增強 (死魚臉增強語氣)
        self.vision_to_audio_attn = CrossModalAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        
        # 新增：諷刺/不一致性偵測 (Incongruity Detection)
        self.incongruity_module = MultimodalIncongruityModule(d_model=d_model)
        
        # 新增核心：TFN 多模態張量融合網路 (取代單純拼接)
        # 用於生成融合 T, A, V 交互高階特徵的 d_model 向度
        self.tfn_fusion = TensorFusionNetwork(d_model=d_model, output_dim=d_model, dropout=dropout)
        
        # 最新架構：MMoE 路由器 (動態分配權重給 TFN, Incongruity, Persona 三大專家)
        self.mmoe_router = FeatureGatingNetwork(d_model=d_model, num_experts=3)
        
        # 3. 序列建模 (Sequence Modeling，捕捉時間依賴 t_<=t)
        # 融合後的特徵: (TFN_Fused, Incongruity, Persona) 共計 3 個 d_model 維度
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 3, 
            nhead=nhead, 
            dim_feedforward=d_model * 3,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        # 4. 決策層 1: Humor Classifier (二元分類)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1) # 輸出幽默 logit (1: 幽默, 0: 非幽默)
        )
        
        # 5. 決策層 2: 結構標籤序列 (Structural Sequence Labeling)
        self.struct_classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(self.STRUCT_TAGS)) # 5 種類別 (O, ST, BR, PL, TL)
        )
        
        # 6. 決策層 3: 技術標籤序列 (Technique Sequence Labeling)
        # 用於檢測 NLP 技術濾鏡: Analogy, Persona, Status Switch
        self.tech_classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(self.TECH_TAGS)) # 4 種類別 (O, Analogy, Persona, StatusSwitch)
        )
        
        # 7. 決策層 4: 二元結構 Context-Punchline 預測器 (Binary Context-Punchline Predictor)
        # 基於 UR-FUNNY 精神：不用生硬的 Pooling，而是用 Punchline 當作 Query，去 Attention 整個 Context Memory
        self.cp_attention = nn.MultiheadAttention(
            embed_dim=d_model * 3, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.cp_classifier = nn.Sequential(
            nn.Linear(d_model * 3 * 2, d_model), # 串聯 Context(Attended) 與 Punchline 特徵
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # 8. 條件隨機場轉換機率矩陣 (CRF-like Transition Matrix)
        # 實踐 P(Y|X) 的轉移項，學習 P(y_i | y_{i-1}) 的時序規律
        # 例如：「Setup 之後通常跟著 Punchline」、「如果是 Bridge 接下來極高機率反轉」
        self.struct_transitions = nn.Parameter(torch.empty(len(self.STRUCT_TAGS), len(self.STRUCT_TAGS)))
        nn.init.uniform_(self.struct_transitions, -0.1, 0.1)
        
    def _attend_context(self, seq_out: torch.Tensor, punchline: torch.Tensor) -> torch.Tensor:
        """
        [引入 UR-FUNNY 概念] Contextual Memory Network 機制
        使用 Punchline (P) 作為 Query 去尋找 Context (C) 中最不協調或最重要的前置資訊。
        """
        batch_size, seq_len, dim = seq_out.shape
        if seq_len <= 1:
            return seq_out[:, 0, :] # 沒有 Context，直接回傳
        
        # Context 作為 Key, Value (去掉最後一個 Token)
        context = seq_out[:, :-1, :] # (B, L-1, dim)
        
        # Punchline 作為 Query
        query = punchline.unsqueeze(1) # (B, 1, dim)
        
        # 注意力讀取 (Attention Read)
        attn_out, _ = self.cp_attention(query=query, key=context, value=context)
        
        return attn_out.squeeze(1) # 回傳 (B, dim) 作為融合了 P 重點的 C 表徵

    def forward(
        self, 
        text_features: torch.Tensor, 
        audio_features: torch.Tensor, 
        vision_features: torch.Tensor,
        persona_features: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向傳播
        
        Args:
            text_features: (batch_size, seq_len, text_dim)
            audio_features: (batch_size, seq_len, audio_dim)
            vision_features: (batch_size, seq_len, vision_dim)
            persona_features: (batch_size, persona_dim)
            src_key_padding_mask: (batch_size, seq_len) 忽略的 pad 遮罩
            
        Returns:
            logits: 用於 Humor 發生預測
            struct_logits: 結構標籤預測 (Setup, Bridge...)
            tech_logits: 技術標籤預測 (Analogy...)
            cp_logits: Context-Punchline 預測
            C_rep: Context 表徵向量 (用於 2025 對比學習 Contrastive Learning)
            P_rep: Punchline 表徵向量 (用於 2025 對比學習 Contrastive Learning)
        """
        batch_size, seq_len, _ = text_features.shape

        # 投影到對齊的潛在空間
        T_h = self.text_proj(text_features)     # (B, L, D)
        A_h = self.audio_proj(audio_features)   # (B, L, D)
        V_h = self.vision_proj(vision_features) # (B, L, D)
        P_h = self.persona_proj(persona_features).unsqueeze(1).expand(-1, seq_len, -1) # (B, L, D)
        
        # 模態間增強 (Cross-modal Interaction) 
        # Text 增強 Audio
        A_enhanced_by_T = self.text_to_audio_attn(query=A_h, key=T_h, value=T_h)
        # Vision 增強 Audio
        A_enhanced_by_V = self.vision_to_audio_attn(query=A_h, key=V_h, value=V_h)
        
        # 提取模態衝突 (諷刺) 特徵
        incongruity_feat = self.incongruity_module(T_h, A_h, V_h)
        
        # 合併增強後的 Audio (Residual connection)
        A_fused = A_h + A_enhanced_by_T + A_enhanced_by_V
        
        # TFN 核心：三階張量外積融合 (取代單純拼接)
        # 產生能夠捕捉 Unimodal, Bimodal 與 Trimodal 交互的高階張量特徵
        tfn_fused_feat = self.tfn_fusion(T_h, A_fused, V_h) # (B, L, D)
        
        # MMoE 動態路由 (取代直接相加或拼接)
        # 讓神經網路根據當下情境自己決定要聽 TFN、聽 Incongruity 還是聽 Persona 的
        fused_features = self.mmoe_router(experts=[tfn_fused_feat, incongruity_feat, P_h]) # (B, L, 3*D)
        
        # 添加時間維度的依賴
        seq_out = self.transformer_encoder(
            fused_features, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 通過各節點分類器 (包含 Transformer 擷取到的時序脈絡)
        logits = self.classifier(seq_out).squeeze(-1) # (B, L)
        
        # (B, L, num_tags)
        struct_logits = self.struct_classifier(seq_out) 
        tech_logits = self.tech_classifier(seq_out)
        
        # 實作 Context-Punchline 二元結構預測 ( UR-FUNNY Contextual Memory)
        # P = 擷取 seq_out[:, -1]
        P_rep = seq_out[:, -1, :]                      # (B, 3*D)
        # C = 使用 P 作為 Query 對歷史進行 Attention
        C_rep = self._attend_context(seq_out, P_rep)   # (B, 3*D)
        
        cp_features = torch.cat([C_rep, P_rep], dim=-1) # (B, 6*D)
        
        cp_logits = self.cp_classifier(cp_features)    # (B, 1)
        
        # 回傳 C_rep 與 P_rep 以支援 2024-2025 SOTA 的 InfoNCE 對比學習 (Contrastive Learning)
        # 訓練時可強迫 C_rep 與 真實 P_rep 在潛在空間拉近/建立反轉映射，並遠離假的 P_rep
        # （在實際 Loss 訓練中，struct_logits 會加上 self.struct_transitions 計算 CRF Loss）
        return logits, struct_logits, tech_logits, cp_logits, C_rep, P_rep


class WeightedCrossEntropyLoss(nn.Module):
    """
    處理極度不平衡的笑聲樣本 (Class Imbalance)
    
    在一場脫口秀中，「笑」的時間點遠少於「沒笑」的時間點，
    這會導致模型偏好預測「不幽默」。利用加權來懲罰漏報 (False Negatives)。
    L = - 1/N * sum [w * y_i * log(y_hat_i) + (1-y_i)*log(1-y_hat_i)]
    """
    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len) 原始模型輸出
            targets: (batch_size, seq_len) 真實標籤 {0, 1}
            mask: (batch_size, seq_len) boolean 遮罩，True 代表有效 token
        """
        pos_weight = self.pos_weight.to(logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets.float(), 
            pos_weight=pos_weight,
            reduction='none'
        )
        
        if mask is not None:
            # 只計算有效序列的 loss
            bce_loss = bce_loss * mask.float()
            return bce_loss.sum() / mask.float().sum().clamp(min=1e-8)
            
        return bce_loss.mean()


class HumorContrastiveLoss(nn.Module):
    """
    2024-2025 頂會趨勢：Setup-Punchline 模態對齊的對比學習 (InfoNCE)
    
    傳統模型只用 CrossEntropy 預測「是不是笑點」，容易淪為背誦特定單字。
    此損失函數會強制計算 Context(鋪陳) 與其專屬的 Punchline(笑點) 在潛在空間的內積。
    目標：
    1. 最大化同一個笑話中 Context 與 Punchline 的互資訊 (互相吸引)。
    2. 同時推開該 Context 與 Batch 內其他「假 Punchline/別人的笑點」的距離 (互相排斥)。
    
    這樣能強迫神經網路學到的不是「死背爆點」，而是「這個情境為何『必然』導致這個反轉」。
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, c_rep: torch.Tensor, p_rep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_rep: (batch_size, dim) 聚合後的 Context 向量
            p_rep: (batch_size, dim) 對應的 Punchline 向量
        Returns:
            InfoNCE 對比損失
        """
        # (Batch, Batch) 的餘弦相似度矩陣
        c_norm = F.normalize(c_rep, p=2, dim=1)
        p_norm = F.normalize(p_rep, p=2, dim=1)
        
        # 計算相似度並依照 Temperature 放縮，以增加難度
        logits = torch.matmul(c_norm, p_norm.transpose(0, 1)) / self.temperature
        
        # 對角線上的是 True Positive (C_i 應該要高度相似 P_i)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # 對稱式 InfoNCE Loss (SimCLR 經典作法)
        loss_c = F.cross_entropy(logits, labels)
        loss_p = F.cross_entropy(logits.transpose(0, 1), labels)
        
        return (loss_c + loss_p) / 2.0
