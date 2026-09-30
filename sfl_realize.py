import torch

def euclidean_proximity_lookup(z_terminal, feature_dict):
    """
    Finds clause realization w* minimizing Euclidean distance ||f_w - z_terminal||_2.
    """
    best_match = None
    min_dist = float('inf')
    for sentence, f_w in feature_dict.items():
        dist = torch.norm(torch.tensor(f_w) - z_terminal, p=2).item()
        if dist < min_dist:
            min_dist = dist
            best_match = sentence
    return best_match, min_dist

if __name__ == "__main__":
    z_terminal = torch.tensor([1.21, 0.44, 1.09, 0.86, 0.91, 0.69, 0.96, 0.29, 1.16])
    
    dict_en = {
        "The institutional committee hereby resolves the policy clause.": [1.20, 0.45, 1.10, 0.85, 0.90, 0.70, 0.95, 0.30, 1.15],
        "Standard procedure dictates immediate review.": [0.50, 0.10, 0.40, 0.30, 0.20, 0.10, 0.60, 0.80, 0.20]
    }
    
    dict_es = {
        "El comité institucional resuelve por la presente la cláusula.": [1.18, 0.42, 1.05, 0.88, 0.93, 0.65, 0.94, 0.27, 1.12]
    }
    
    dict_fr = {
        "Le comité institutionnel résout par la presente la clause.": [1.19, 0.43, 1.07, 0.87, 0.92, 0.67, 0.95, 0.28, 1.14]
    }
    
    m_en, d_en = euclidean_proximity_lookup(z_terminal, dict_en)
    m_es, d_es = euclidean_proximity_lookup(z_terminal, dict_es)
    m_fr, d_fr = euclidean_proximity_lookup(z_terminal, dict_fr)
    
    print(f"EN: '{m_en}' (d = {d_en:.4f})")
    print(f"ES: '{m_es}' (d = {d_es:.4f})")
    print(f"FR: '{m_fr}' (d = {d_fr:.4f})")
