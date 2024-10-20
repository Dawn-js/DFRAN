# Advancing Chinese Travel Sentiment Analysis: A Novel Dataset and DFRAN Approach for Missing Modalities
Authors: Bo Sun , Turdi Tohti , Dongfang Han , Yi Liang , Zicheng Zuo , Yuanyuan Liao , Qingwen Yang

With the proliferation of user-generated travel reviews, sentiment analysis in this domain has become crucial. Despite advancements in integrating textual and visual modalities, high-quality sentiment analysis datasets for Chinese travel reviews remain limited, and the challenge of missing modalities is often overlooked. To address these issues, we introduce the Chinese Travel Review Sentiment (CTRS) dataset, comprising 51,306 image-text pairs collected from the Trip platform. Additionally, we propose the Distribution Feature Recovery and Auxiliary Networks (DFRAN) approach to handle missing modalities. DFRAN utilizes a flow-based generative network to learn the distribution of available modalities and recover missing information. Auxiliary networks extract modality-specific features, enriching the representations learned by the image-text fusion network and mitigating the impact of discrepancies between generated and real data. Extensive experiments on benchmark datasets demonstrate the effectiveness of the CTRS dataset and the superiority of DFRAN in scenarios with missing modalities, advancing the field of image-text sentiment analysis for Chinese travel reviews.The source code and dataset are available from https://github.com/Dawn-js/DFRAN.

# Prerequisites:
Python 3.9
Pytorch 2.0.0
CUDA 11.8
# Datasets:

# Training:
