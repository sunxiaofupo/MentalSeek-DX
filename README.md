# MentalSeek-Dx

**Progressive Hypothetico-Deductive Reasoning for Real-world Psychiatric Diagnosis**

## Introduction

MentalSeek-Dx is a medical-specialized large language model (LLM) designed to address real-world psychiatric diagnosis, with a focus on aligning AI reasoning with clinical practices. Mental health disorders are a growing global public health concern. While LLMs are promising for psychiatric assessment, existing benchmarks often lack real-world complexity and detailed diagnostic supervision, which limits their clinical applicability.

To address this, we introduce **MentalDx Bench** — the first benchmark targeting disorder-level psychiatric diagnosis in real clinical settings. The benchmark consists of 712 de-identified electronic health records, each annotated by board-certified psychiatrists using ICD-11 guidelines, covering 76 disorders across 16 diagnostic categories.

Evaluation of 18 LLMs identified a "paradigm misalignment": while many models perform well at broad diagnostic categorization, they struggle with fine-grained, disorder-level diagnosis, revealing a gap between pattern recognition and rigorous clinical reasoning.

MentalSeek-Dx bridges this gap by internalizing clinical hypothetico-deductive reasoning. It is trained using supervised trajectory construction and curriculum-based reinforcement learning to more faithfully simulate a psychiatrist’s diagnostic process.

**Key Highlights:**
- State-of-the-art (SOTA) performance on the MentalDx Bench with only 14B parameters
- Deep integration of clinical reasoning, moving beyond pattern-based diagnosis
- Establishes a practical, trustworthy foundation for psychiatric diagnosis with LLMs

For more details, usage instructions, and evaluation steps, see below.
