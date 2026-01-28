---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

# Biography

<span class='anchor' id='about-me'></span>

I am a PhD student in the Institute for Language, Cognition and Computation at the School of Informatics, University of Edinburgh. My doctoral research is supervised by Mark Steedman and Shay Cohen.

My work focuses on semantic parsing, information extraction, retrieval-augmented generation, textual entailment, and natural language understanding, with a particular emphasis on mitigating hallucinations and improving the inference capabilities of large language models. More recently, I have also been exploring multimodal reasoning tasks.

My research has been published in top-tier AI and NLP venues, including ACL, EMNLP, EACL, NeurIPS, and COLING.


# 🔥 Recent News in 2025
- *2026.01*: &nbsp;🎉🎉 Our work "*RAGBoost: Efficient Retrieval-Augmented Generation with Accuracy-Preserving Context Reuse*" has been accepted in MLSys 2026.
- *2025.09*: &nbsp;🎉🎉 Our work "*MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly*" has been accepted in NeurIPS 2025 Spotlight.
- *2025.08*: &nbsp;🎉🎉 Our work "*S2LPP: Small-to-Large Prompt Prediction across LLMs*" has been accepted in EMNLP 2025.
- *2025.05*: &nbsp;🎉🎉 Our work "*Neutralizing Bias in LLM Reasoning using Entailment Graphs*" has been accepted in ACL 2025.
- *2025.01*: &nbsp;🎉🎉 Our work "*Empirical Study on Data Attributes Insufficiency of Evaluation Benchmarks for LLMs*" has been accepted in COLING 2025.


# 📝 Publications 
### Citations (until 09/12/2025):  330

### [RAGBoost: Efficient Retrieval-Augmented Generation with Accuracy-Preserving Context Reuse](https://arxiv.org/abs/2511.03475)
> Present an efficient RAG system that maximizes cache reuse via accuracy-preserving context reuse, detecting overlaps across sessions and reusing contexts while maintaining reasoning accuracy. (*MLSys 2026*)

### [MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](https://neurips.cc/virtual/2025/loc/san-diego/poster/121768)
> Design a benchmark to evaluate the long-context capabilities of large vision-language models across 46 VLMs and five task types, revealing key challenges in long-context reasoning.(*NeurIPS 2025*)

### [S2LPP: Small-to-Large Prompt Prediction across LLMs](https://aclanthology.org/2025.findings-emnlp.483/)
> Present consistent prompt preference across LLMs in QA, NLI, RAG and Reasoning tasks. We further propose a lightweight approach that uses smaller models for cost-efficient prompt selection in open-domain questions and reasoning tasks.
(*EMNLP 2025*)

### [Neutralizing Bias in LLM Reasoning using Entailment Graphs](https://aclanthology.org/2025.findings-acl.705/)
> Proposed an unsupervised framework to generate counterfactual reasoning data to train LLMs, effectively reducing hallucinations and memorization biases in reasoning and QA tasks while enhancing LLMs' inferential capabilities.
(*ACL 2025*)

### [Empirical Study on Data Attributes Insufficiency of Evaluation Benchmarks for LLMs](https://aclanthology.org/2025.coling-main.403/)
> Proposed an automated framework to evaluate data diversity, redundancy, and difficulty in LLM benchmarks.
(*COLING 2025*)

### [Explicit Inductive Inference using Large Language Models](https://aclanthology.org/2024.findings-emnlp.926/)
> Proposed an explicit inductive pipeline leveraging the attestation bias of LLMs to enhance inference robustness.
(*EMNLP 2024*)

### [Sources of Hallucination by Large Language Models on Inference Tasks](https://aclanthology.org/2023.findings-emnlp.182/)
> Identify two biases originating from LLMs pretraining and present that these are major sources of hallucination in LLMs.
(*EMNLP 2023*)

### [Complementary Roles of Inference and Language Models in QA](https://aclanthology.org/2023.pandl-1.8/)
> Developed RAG agents for open-domain QA by extracting knowledge graphs and integrating large language models, enhancing explainability and precision. We further proposed unsupervised textual entailment extraction methods to mitigate the sparsity of knowledge graphs. (*PANDL@EMNLP 2023*)

### [LLMs are Frequency Pattern Learners in Natural Language Inference](https://arxiv.org/abs/2505.21011)
> Identify a correlation between frequency patterns and semantic generalization gradient, providing explanations for the source of LLMs' inferential capability. Building on these insights, we propose an efficient training strategy that uses small data subsets. (*Submitted Anonymous*)

### [Sentence-Level Soft Attestation Bias of LLMs](https://openreview.net/pdf?id=bmF7kkjj8B)
> Proposed soft attestation to measure attestation biases in LLMs during NLI tasks, showing that LLMs often prefer factual hypotheses over true entailment, though this bias can also be used to improve performance. (*Submitted Anonymous*)


# 📖 Educations
- *2021.04 -  now*, I feel very fortunate to be supervised by Prof. **Mark Steedman** and **Shay Cohen** during my PhD life in University of Edinburgh!

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Mark Steedman</div><img src='images/Mark.jpg' alt="sym" width="50%"></div></div>
<div class='paper-box-text' markdown="1">

[Mark Steedman](https://homepages.inf.ed.ac.uk/steedman/)

ACL Lifetime Achievement Award, Fellow of the American Association for Artificial Intelligence, the British Academy, the Royal Society of Edinburgh, the Association for Computational Linguistics, and the Cognitive Science Society.

**Institute:** Institute for Language, Cognition and ComputationSchool of Informatics, University of Edinburgh. <strong><span class='show_paper_citations' data='DhtAFkwAAAAJ:ALROH1vI_8AC'></span></strong> 
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Shay Cohen</div><img src='images/Shay.jpg' alt="sym" width="50%"></div></div>
<div class='paper-box-text' markdown="1">

[Shay Cohen](https://homepages.inf.ed.ac.uk/scohen/)

Reader at the University of Edinburgh (School of Informatics). 

**Institute:** Institute for Language, Cognition and ComputationSchool of Informatics, University of Edinburgh. <strong><span class='show_paper_citations' data='DhtAFkwAAAAJ:ALROH1vI_8AC'></span></strong>
</div>
</div>

- *2016.09 - 2019.06*, Institute of Computing Technology, Chinese Academy of Sciences. Master.


# 💻 Work Experience
- ### *2019.09 - 2020.02*, Huawei Noah’s Ark Lab (Researcher), China.
> • Combining vision with speech. We propose a model that uses lip images from videos to enhance the quality of phone calls, which has since been successfully integrated into Huawei devices.<br>
> • Use Generative Flow (Glow) algorithm for speaker recognition, enhancing the quality of phone contact。<br>
> • Implemented an image caption generation system for photo tools.<br>  
> Award: Future Star Award of Huawei Noah’s Ark Lab.

- ### *2018.01 - 2019.01*, E Fund Management Co., Ltd. (Internship), China.
> Developed a high-frequency quantitative trading system by combining Temporal Convolutional Neural Networks (TCNNs) with reinforcement learning.

- ### *2017.01 - 2018.01*, University of Chinese Academy of Science (Teaching Assistant), China.
> Teaching reinforcement learning and graph learning algorithms in University of Chinese Academy of Science. I am responsible for teaching the fundamental principles and concepts of algorithms.


# 📸 Photography
Photography is my way of seeing the world—here are some of my recent shots.

<style>
.photo-list {
  display: block;
}

/* One full-width image */
.photo-item img {
  width: 100%;
  display: block;
  margin-bottom: 20px;
}

/* Row of thin images */
.half {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

/* Individual image + title block */
.half-item {
  flex: 1;
  text-align: center;
}

.half-item img {
  width: 100%;
  border-radius: 6px;
}

.badge {
  font-weight: bold;
  margin-bottom: 8px;
  display: block;
}
</style>

<div class="photo-list">

  <!-- FULL-WIDTH IMAGE -->
  <div class="photo-item">
    <div class="badge">Foggy Night at Edinburgh</div>
    <img src="images/Edinburgh Fog Street.jpg" alt="Edinburgh Foggy Night">
  </div>

  <!-- THREE IMAGES IN ONE ROW -->
  <div class="half">
    <div class="half-item">
      <div class="badge">Edinburgh Night</div>
      <img src="images/Edinburgh Night.jpg" alt="Edinburgh Night">
    </div>
    <div class="half-item">
      <div class="badge">Edinburgh Winter</div>
      <img src="images/Edinburgh Winter.jpg" alt="Edinburgh Winter">
    </div>
    <div class="half-item">
      <div class="badge">I am a Rock Star</div>
      <img src="images/Rock Star.jpg" alt="I am a Rock Star">
    </div>
  </div>
 
</div>

{% include effects.html %}