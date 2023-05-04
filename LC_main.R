rm(list = ls())

#N      the full data size
#p      the dimensionality of covariates X
#n0     the sample size in pliot study 
#n      the subsample size  
#delta  the imblanced degree


library(MASS)
library(e1071)
library(ggplot2)
library(grid)
library(latex2exp)
library(mnormt)
library(glmnet)
library(WeightSVM)
library(flexclust)
library(caret)
library(LiblineaR)
library(e1071)
library(sgd)

setwd("C:/Users/STAT/hyx/SVM Sampling/online/code_submit")
source("LC_function.R",local=TRUE)


N <- 10^5
p <-  8    
BB <- 500                       #simulation times
scale <- FALSE                  #scale data or not in SVM function
nfold <- 10                     #k fold cross-validation 
kernel <- "linear"              #the kernel used in SVM  #radial
range <- 10^(seq(-5,0,0.5))     #the tuning grid range 
replace <-TRUE                  #sampling with replacement
packs <- c("WeightSVM","flexclust","mnormt","MASS","caret","LiblineaR")


#------------------------------------------------------------------------------------------
###----Generate data------

case <- "unif";delta <- 0.2
#case <- "normMIX333"; delta <- 0.5
#case <- "T3"; delta <- 0.5
#case <- "T3MIX222"; delta <- 0.5


N1 <- delta*N
N2 <- N-N1
n0 <- 500

set.seed(2020)
#generate data
data <- Gendata(N,N1,N2,p,case)
df_train <- data$df_train
df_test <- data$df_test

#data
X <- df_train$X 
XC <- cbind(rep(1,nrow(X)),X)    #with intercept
Y <- df_train$class                #factor
Y_num <- as.numeric(as.character(Y))  #numeric       #Y needs to product with other numeric; mode numeric
dat_train <- list(X=X,XC=XC,class=Y,Y_num=Y_num)

X_test <- df_test$X
XC_test <- cbind(rep(1,nrow(X_test)),X_test)
Y_test <- df_test$class                                #factor
Y_test_num <- as.numeric(as.character(Y_test))         #fit model to predict
dat_test <- list(X=X_test, class=Y_test,XC = XC_test, Y_num=Y_test_num)

#result on full data (fix results if N does not change)
weight_full <- rep(1,N)  #full data with unweighted weight
#comparison full data
#fit_SVM <- SVM_LiblineaR(dat_train,dat_test,range,scale,nfold,kernel,weight=weight_full)  #model SVM: construct model on trianing data
#comparison acc and mse
fit_SVM <- SVM_full(dat_train,dat_test,range,scale,nfold,kernel,weight=weight_full)   
acc_SVM <- fit_SVM$accuracy                    #accuracy
error_SVM <- fit_SVM$miserror                  #misclassification error rate in test data
nSV_SVM <- fit_SVM$nSV                         #number of support vectors
beta_SVM <- fit_SVM$beta
cost_SVM <- fit_SVM$cost
temperror_SVM <- fit_SVM$temperror
time_SVM_train <- fit_SVM$time_train
time_SVM_tune <- fit_SVM$time_tune
time_SVM <- time_SVM_train + time_SVM_tune
result_SVM <- list(acc_SVM = acc_SVM, time_SVM = time_SVM, cost_SVM = cost_SVM,beta_SVM = beta_SVM,
                   time_SVM_train = time_SVM_train,time_SVM_tune=time_SVM_tune)
result_SVM


###-----n vs accuracy,MSE-------
library(parallel)
library(foreach)
library(iterators)
library(doParallel)
cl <- makeCluster(25)
registerDoParallel(cl)

BB <- 500 
n0 <- 500
n <- c(50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000);length(n)

result_mean <- matrix(0,length(n),23)
colnames(result_mean) <- c("acc_mMSE","acc_mVc","acc_SVM","acc_SVM_uni","time_mMSE","time_mVc","time_SVM","time_SVM_uni","time_mMSE_train","time_mVc_train","time_SVM_train","time_SVM_uni_train","time_mMSE_tune","time_mVc_tune","time_SVM_tune","time_SVM_uni_tune","cost_mMSE","cost_mVc","cost_SVM","cost_SVM_uni","MSE_mMSE","MSE_mVc","MSE_SVM_uni")
rownames(result_mean) <- n
result_sd <- matrix(0, length(n), 23)
colnames(result_sd) <- c("acc_mMSE","acc_mVc","acc_SVM","acc_SVM_uni","time_mMSE","time_mVc","time_SVM","time_SVM_uni","time_mMSE_train","time_mVc_train","time_SVM_train","time_SVM_uni_train","time_mMSE_tune","time_mVc_tune","time_SVM_tune","time_SVM_uni_tune","cost_mMSE","cost_mVc","cost_SVM","cost_SVM_uni","MSE_mMSE","MSE_mVc","MSE_SVM_uni")
rownames(result_sd) <- n
temp <- array(0, dim=c(BB,23,length(n)))


for (j in 1:length(n)){
  tic <- proc.time()
  
  tem <- foreach(i=1:BB,.combine ="rbind",.packages=packs,.errorhandling="remove")%dopar%
    OptimalSamp(result_SVM,dat_train,dat_test,N,delta,p,n0,n[j],range,kernel,replace,scale,case,i)
  result_mean[j,] <- apply(tem,2,mean)
  result_sd[j,] <- apply(tem,2,sd)
  
  tic2 <- proc.time()-tic
  
  print(paste("Sam.",N,"-n.",n[j],"-time:",tic2[3],sep=""))
}
stopCluster(cl)

result_mean
write.csv(result_mean,file='result_mean.csv')
write.csv(result_sd, file = "result_sd.csv")

result_mean[,c(1:4)]   #acc:1-4;
result_mean[,c(21:23)] #MSE:21-23
result_mean[,c(5:8)]   #full time:5-8

#plot n vs MSE
methods <- c("LC-A","LC-L","LC-UNIF");length(methods)
plane_index <-  c(21,22,23)  
df_plane <- data.frame(plane = as.numeric(unlist(result_mean[,plane_index])),
                       n = rep(n,length(methods)),
                       method = factor(rep(methods,each=length(n)),levels=methods),
                       structure= rep("im-Uniform",length(methods)*length(n)))
wd=0.5
library(ggplot2)
library(grid)
library(latex2exp)


fig_plane <- ggplot(df_plane)+
  geom_point(aes(x=n, y=plane,color =method,shape=method),size=2.5*wd)+
  geom_line(aes(x=n, y=plane,color =method,linetype=method),size=wd)+
  theme(legend.position="top")+
  xlab(TeX('$\\n$'))+
  ylab("MSE")+
  theme(plot.title = element_text(hjust = 0.5))
fig_plane 


#plot n vs accuracy
methods <- (c("LC-A","LC-L","SVM-FULL","LC-UNIF"));length(methods)
acc_index <- c(1,2,3,4)
df_acc <- data.frame(acc = as.numeric(unlist(cbind(result_mean[,acc_index]))),
                     n = rep(n,length(methods)),
                     method = factor(rep(methods,each=length(n)),levels=methods),
                     structure= rep("im-Uniform",length(methods)*length(n)))
fig_acc <- ggplot(df_acc)+
  geom_point(aes(x=n, y=100*acc,color =method, shape = method),size=2.5*wd)+
  geom_line(aes(x=n, y=100*acc,color =method,linetype=method),size=wd)+
  theme(legend.position="top")+
  xlab(TeX('$\\n$'))+
  ylab("Prediction accuracy")+
  theme(plot.title = element_text(hjust = 0.5)) 
fig_acc 


###----full size N-------------


n0 <- 500
n <- 1000
N <- c(10^3,10^3,10^4,10^5,10^6,10^7)
delta <- 0.2
N1 <- delta*N
N2 <- N-N1
case <- "unif"


result_full <- matrix(0,length(N),12)
colnames(result_full) <- c("time_mMSE","time_mVc","time_SVM","time_SVM_uni","time_mMSE_train","time_mVc_train","time_SVM_train","time_SVM_uni_train","time_mMSE_tune","time_mVc_tune","time_SVM_tune","time_SVM_uni_tune")
rownames(result_full) <- N
temp <- result_full

for (j in 1:length(N)){
  i <- 1
  set.seed(2020+i)
  #generate data
  data <- Gendata(N[j],N1[j],N2[j],p,case)
  df_train <- data$df_train
  df_test <- data$df_test
  
  #data
  X <- df_train$X
  XC <- cbind(rep(1,nrow(X)),X)  #intercept
  Y <- df_train$class                #factor
  Y_num <- as.numeric(as.character(Y))  #numeric        
  dat_train <- list(X=X,XC=XC,class=Y,Y_num=Y_num)
  
  X_test <- df_test$X
  XC_test <- cbind(rep(1,nrow(X_test)),X_test)
  Y_test <- df_test$class                      #factor
  Y_test_num <- as.numeric(as.character(Y_test))          
  dat_test <- list(X=X_test, class=Y_test,XC = XC_test, Y_num=Y_test_num)
  
  #result on full data 
  weight_full <- rep(1,N[j])  #full data ,unweighted
  fit_SVM <- SVM_LiblineaR(dat_train,dat_test,range,scale,nfold,kernel,weight=weight_full)  #model SVM: construct model on trianing data
  acc_SVM <- fit_SVM$accuracy                    #accuracy
  error_SVM <- fit_SVM$miserror                  #misclassification error rate in test data
  beta_SVM <- fit_SVM$beta
  cost_SVM <- fit_SVM$cost
  temperror <- fit_SVM$temperror
  time_SVM_train <- fit_SVM$time_train
  time_SVM_tune <- fit_SVM$time_tune
  time_SVM <- time_SVM_train + time_SVM_tune
  result_SVM <- list(acc_SVM = acc_SVM, time_SVM = time_SVM, cost_SVM = cost_SVM,beta_SVM = beta_SVM,
                     time_SVM_train = time_SVM_train,time_SVM_tune=time_SVM_tune)
  
  tic = proc.time()
  temp[j,] <- OptimalSamp(result_SVM,dat_train,dat_test,N[j],delta,p,n0,n,range,kernel,replace,scale,case,i)[5:16]
  
  tic2 = proc.time()-tic
  
  result_full[j,]=temp[j,]
  print(paste("Sam.",N[j],"-n0.",n0,"-n.",n,"-time:",tic2[3],sep=""))
}

result_full
write.csv(result_full,file='result_full.csv')
result_full[,c(1:4)]


methods <- c("LC-A","LC-L","SVM-FULL","LC-UNIF");length(methods)
time_index <- c(1,2,3,4)  #full time 



df_unif_time <- data.frame(time = as.numeric(unlist(result_full[,time_index])),
                           N = rep(log(N,10),length(methods)),
                           method = factor(rep(methods,each=length(N)),levels=methods),
                           structure= rep("im-Uniform",(length(methods))*length(N)))
df_unif_time$time <- log(df_unif_time$time,10)

wd <- 0.5
library(ggplot2)
library(grid)
library(latex2exp)

fig_unif_time <- ggplot(df_unif_time)+
  geom_point(aes(x=N, y=time,color =method,shape=method),size=2.5*wd)+
  geom_line(aes(x=N, y=time,color =method,linetype =method),size=wd)+
  theme(legend.position="top")+
  xlab(TeX('$\\logN$'))+
  ylab("log Time/s")+
  theme(plot.title = element_text(hjust = 0.5)) 
fig_unif_time <- fig_unif_time + 
  facet_grid(.~ structure)+theme(text=element_text(family="serif"))
fig_unif_time