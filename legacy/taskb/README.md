# CodeBERT + Stylometric + Entropy-Biased Attention Architecture

```
INPUT: Code Snippet
"def func(x): if x > 0: return x*2; return 0"
                    │
                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    INPUT PROCESSING                          │
    └─────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌─────────────────┐    ┌─────────────────────┐
│   TOKENIZATION  │    │  STYLOMETRIC        │
│                 │    │  FEATURE EXTRACTION │
│ CodeBERT        │    │                     │
│ Tokenizer       │    │ • Lines of Code     │
│                 │    │ • Complexity        │
│ [CLS] def func  │    │ • Halstead Metrics  │
│ ( x ) : if x    │    │ • Comments Count    │
│ > 0 : return    │    │ • Function Count    │
│ x * 2 ; return  │    │                     │
│ 0 [SEP]         │    │ [0.2, 1.5, 3.0,    │
│                 │    │  0.1, 2.0, ...]     │
└─────────────────┘    └─────────────────────┘
        │                       │
        │                       │ (20-dim vector)
        ▼                       │
┌─────────────────────────────────────────────────────────────┐   │
│                    CODEBERT ENCODER                         │   │
│                                                             │   │
│  Token Embeddings + Position Embeddings                    │   │
│                    │                                        │   │
│                    ▼                                        │   │
│  ┌─────────────────────────────────────────────────────┐   │   │
│  │              TRANSFORMER LAYERS                     │   │   │
│  │                                                     │   │   │
│  │  Layer 1: Multi-Head Self-Attention + FFN          │   │   │
│  │  Layer 2: Multi-Head Self-Attention + FFN          │   │   │
│  │  ...                                                │   │   │
│  │  Layer 12: Multi-Head Self-Attention + FFN         │   │   │
│  └─────────────────────────────────────────────────────┘   │   │
│                    │                                        │   │
│                    ▼                                        │   │
│  Hidden States: [batch, seq_len, 768]                      │   │
│  [[h₁₁, h₁₂, ..., h₁₇₆₈],                                  │   │
│   [h₂₁, h₂₂, ..., h₂₇₆₈],                                  │   │
│   ...                                                       │   │
│   [hₙ₁, hₙ₂, ..., hₙ₇₆₈]]                                  │   │
└─────────────────────────────────────────────────────────────┘   │
        │                                                         │
        ▼                                                         │
┌─────────────────────────────────────────────────────────────┐   │
│                 ENTROPY COMPUTATION                         │   │
│                                                             │   │
│  For each token position i:                                 │   │
│  entropy[i] = -Σ(p * log(p))                              │   │
│                                                             │   │
│  Hidden States     →    Token Entropy                      │   │
│  [h₁₁, h₁₂, ..., h₁₇₆₈] → entropy₁ = 2.3 (high)           │   │
│  [h₂₁, h₂₂, ..., h₂₇₆₈] → entropy₂ = 0.8 (low)            │   │
│  [h₃₁, h₃₂, ..., h₃₇₆₈] → entropy₃ = 1.9 (medium)         │   │
│  ...                                                        │   │
│                                                             │   │
│  Entropy Vector: [2.3, 0.8, 1.9, 1.2, 2.1, ...]         │   │
└─────────────────────────────────────────────────────────────┘   │
        │                                                         │
        ▼                                                         │
┌─────────────────────────────────────────────────────────────┐   │
│            ENTROPY-WEIGHTED SELF-ATTENTION                 │   │
│                                                             │   │
│  Step 1: Compute Q, K, V projections                       │   │
│  Q = Hidden × W_Q,  K = Hidden × W_K,  V = Hidden × W_V    │   │
│                                                             │   │
│  Step 2: Compute attention scores                          │   │
│  Scores = (Q × K^T) / √d_k                                 │   │
│                                                             │   │
│  Step 3: Add entropy bias                                  │   │
│  Biased_Scores = Scores + α × Entropy_Bias                 │   │
│                                                             │   │
│  ┌─────────────────────────────────────────────────────┐   │   │
│  │        Attention Score Matrix (Before Bias)        │   │   │
│  │                                                     │   │   │
│  │      tok1  tok2  tok3  tok4  tok5                   │   │   │
│  │ tok1 [0.2  0.1   0.3   0.2   0.2]                  │   │   │
│  │ tok2 [0.1  0.4   0.2   0.1   0.2]                  │   │   │
│  │ tok3 [0.3  0.2   0.1   0.3   0.1] ← Complex token  │   │   │
│  │ tok4 [0.2  0.1   0.3   0.3   0.1]                  │   │   │
│  │ tok5 [0.2  0.2   0.1   0.1   0.4]                  │   │   │
│  └─────────────────────────────────────────────────────┘   │   │
│                    │                                        │   │
│                    ▼ (Add entropy bias)                     │   │
│  ┌─────────────────────────────────────────────────────┐   │   │
│  │        Attention Score Matrix (After Bias)         │   │   │
│  │                                                     │   │   │
│  │      tok1  tok2  tok3  tok4  tok5                   │   │   │
│  │ tok1 [0.4  0.2   0.5   0.3   0.3] ← Higher scores  │   │   │
│  │ tok2 [0.2  0.5   0.3   0.2   0.3]                  │   │   │
│  │ tok3 [0.5  0.4   0.3   0.5   0.3] ← Much higher!   │   │   │
│  │ tok4 [0.3  0.2   0.4   0.4   0.2]                  │   │   │
│  │ tok5 [0.3  0.3   0.2   0.2   0.5]                  │   │   │
│  └─────────────────────────────────────────────────────┘   │   │
│                                                             │   │
│  Step 4: Apply softmax and compute weighted output         │   │
│  Attention_Weights = softmax(Biased_Scores)                │   │
│  Attention_Output = Attention_Weights × V                  │   │
└─────────────────────────────────────────────────────────────┘   │
        │                                                         │
        ▼                                                         │
┌─────────────────────────────────────────────────────────────┐   │
│                    POOLING LAYER                           │   │
│                                                             │   │
│  Attention_Output: [batch, seq_len, 768]                   │   │
│                    │                                        │   │
│                    ▼ (Mean pooling)                         │   │
│  Pooled_Output: [batch, 768]                               │   │
│                                                             │   │
│  Represents the entire code snippet as a single vector     │   │
│  with entropy-weighted attention applied                   │   │
└─────────────────────────────────────────────────────────────┘   │
        │                                                         │
        │                                                         │
        │                 ┌───────────────────────────────────────┘
        │                 │
        ▼                 ▼
┌─────────────────┐    ┌─────────────────────┐
│   CODE BRANCH   │    │  STYLOMETRIC BRANCH │
│                 │    │                     │
│ Pooled Output   │    │ Stylometric Vector  │
│ [768-dim]       │    │ [20-dim]            │
│                 │    │         │           │
│                 │    │         ▼           │
│                 │    │ Linear Projection   │
│                 │    │ [20] → [768]        │
│                 │    │                     │
│                 │    │ Projected Stylo     │
│                 │    │ [768-dim]           │
└─────────────────┘    └─────────────────────┘
        │                       │
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │      CONCATENATION      │
        │                         │
        │ [Code Features (768) +  │
        │  Stylo Features (768)]  │
        │                         │
        │ Combined: [1536-dim]    │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │       DROPOUT           │
        │      (p = 0.2)          │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │    CLASSIFICATION       │
        │       LAYER             │
        │                         │
        │ Linear: [1536] → [11]   │
        │                         │
        │ Output Logits:          │
        │ [human, deepseek, qwen, │
        │  01-ai, bigcode, gemma, │
        │  phi, meta-llama,       │
        │  ibm-granite, mistral,  │
        │  openai]                │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │      AUXILIARY          │
        │   STYLOMETRIC LOSS      │
        │                         │
        │ Pooled Output → Linear  │
        │ [768] → [20]            │
        │                         │
        │ Predicted Stylo Features│
        │ MSE Loss with original  │
        │ stylometric features    │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │      FINAL LOSS         │
        │                         │
        │ Total Loss =            │
        │ CrossEntropy(logits) +  │
        │ 0.3 × MSE(stylo_pred)   │
        └─────────────────────────┘

```

## Key Components Explained:

### 1. **Dual Input Processing**
- **Code Path**: Tokenization → CodeBERT encoding
- **Stylometric Path**: Feature extraction (complexity, metrics, etc.)

### 2. **Entropy-Biased Attention Mechanism**
- Computes token-level uncertainty (entropy)
- Adds entropy bias to attention scores **before** softmax
- High entropy tokens get higher attention weights
- Results in focused attention on complex/uncertain code patterns

### 3. **Feature Fusion**
- Code features (entropy-weighted) + Stylometric features
- Both projected to same dimension and concatenated
- Creates rich representation combining semantic and stylistic information

### 4. **Multi-Task Learning**
- Primary task: Authorship classification
- Auxiliary task: Stylometric feature regression
- Helps model learn better stylistic representations

### 5. **Attention Flow Example**
```
Token: "if x > 0 and y < 10:"  (Complex conditional)
├─ High entropy (2.3) → High attention bias
├─ Gets more attention weight in final representation
└─ Model focuses on this discriminative pattern

Token: "return 0"  (Simple statement)  
├─ Low entropy (0.8) → Low attention bias
├─ Gets standard attention weight
└─ Less emphasis in final representation
```

This architecture allows the model to automatically discover and focus on the most discriminative code patterns for authorship detection, combining both semantic understanding (CodeBERT) and explicit stylistic features (stylometric) with uncertainty-guided attention (entropy).