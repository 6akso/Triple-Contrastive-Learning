##Our paper was submitted to the journal INFORMATION SCIENCES in November 2023. It went through one major revision and two minor revisions before being accepted by the editor in June 2024. Unfortunately, the journal INFORMATION SCIENCES encountered issues in June. To meet the needs of our team, we had to resubmit to another journal.

##This is the complete code for the paper "Triple Contrastive Learning Representation Boosting for Supervised Multiclass Tasks," including experiments in both computer vision (CV) and natural language processing (NLP). You can run the code directly to achieve good results. I have also uploaded models trained with the baseline CE+SCL and our improved method TSI-CL.

##Please note that in our code implementation, to enforce the third-order constraint between negative samples with different labels, we have two forms of the loss function. The second form uses tensors for loss calculation to reduce training time. In this form, the constant term in the denominator of the loss function changes from the number of negative samples for anchor 𝑥 to the number of pairwise dot product calculations between negative samples with different labels.

##Since all the experiments mentioned in the paper were completed a year ago, and the code was overly complex during the experiments, I have reorganized it. However, this might also introduce some errors. Although we ensured the code ran correctly on our devices before submission, if you encounter any errors while running the code, please contact us to resolve them together.

##Responsible: Xianshuai Li Email:1849667739@qq.com
