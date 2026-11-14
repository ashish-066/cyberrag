# Third-Party Notices & Data Attribution

This project incorporates data, frameworks, and datasets from third-party sources under their respective licenses:

---

## 1. AnnoCTR Dataset (Bosch Research)

Portions of the entity annotations and token sequences used for benchmark evaluation are derived from the **AnnoCTR** corpus created by Bosch Research.

* **Citation**:  
  *AnnoCTR: A Dataset for Cyber Threat Intelligence Named Entity Recognition and Normalization*,  
  Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024).
* **License**: Creative Commons Attribution-ShareAlike 4.0 International (**CC BY-SA 4.0**)  
  [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)
* **Attribution Notice**: In accordance with the CC BY-SA 4.0 license, you must give appropriate credit, provide a link to the license, and indicate if changes were made. Changes made in this repository include text reconstruction from token lists, refanging of network observables, and sentence/paragraph indexing for retrieval benchmarks.

---

## 2. MITRE ATT&CK® Framework

This platform incorporates structured cybersecurity ontology data derived from the MITRE ATT&CK® enterprise framework.

* **Source**: The MITRE Corporation  
  [https://attack.mitre.org/](https://attack.mitre.org/)
* **Terms of Use**: © 2024 The MITRE Corporation. This work is reproduced and distributed with the permission of The MITRE Corporation under the MITRE ATT&CK® Terms of Use.

---

## 3. Pre-Trained Transformer Models

The system interfaces with open-source pre-trained weights hosted on Hugging Face:

* **Sentence-Transformers**: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0 License)
* **DeBERTa-v3 Cross-Encoder**: `cross-encoder/nli-deberta-v3-base` (Apache-2.0 License)
