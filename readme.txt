################################################################
#Leverage classifier: another look at support vector machine
################################################################


LC_function.R: includes the main functions of our optimal subsampling procedure.


Lc_main.R: 

1. Generate the full dataset under different scenarios. 
           
2. A simulation comparison of LC-A, LC-L, LC-UNIF and SVM-UNIF with different subsample size n 
when fixing full sample size N. The results are shown in our paper Figure 3, Figure 4, and Figure 6. 

3. A simulation comparison of computing time of LC-A, LC-L, LC-UNIF and SVM-UNIF with different full sample size N when fixing subsample size n. The result is presented in our paper Table 1. (Different computing environments may result in slightly different computing times).