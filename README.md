# LGFMDA
LGFMDA：
Code and Datasets for "LGFMDA: miRNA-disease association prediction with local and global feature representation learning"

Requirements：
	python == 3.6
	torch == 1.9.0+cu111 
	numpy == 1.19.5
	dgl-cu111 == 0.6.1

Data：
	D_GIP3.2: Gaussian interaction profile kernel similarity of diseases
	disease semantic similarity_3.2: semantic similarity of diseases
	M_GIP3.2: Gaussian interaction profile kernel similarity of miRNAs
	SeqSim3.2: miRNA sequence similarity
	miRNA fuctional similarity_3.2: miRNA fuctional similarity

Code：
	utils_new: Data reading and construction of attribute bipartite graph
	model_gt_fang_hmdd32_lpe: The overall implementation code of the LGFMDA
	train2_hmdd32_lpe: Training of the model
