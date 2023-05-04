##############################################################
#leverage classifier: another look at support vector machine
##############################################################

#generate data
Gendata <- function(N,N1,N2,p,case){
  
  #simulation
  if(p>2){
    if (case == "unif"){
      X1_train <- matrix(0,N1,p)
      X2_train <- matrix(0,N2,p)
      for (j in 1:p){
        X1_train[,j] <- runif(N1,0,1)   #class 1
        X2_train[,j] <- runif(N2,0.3,1.3)  #class 2
      }
      X_train <- rbind(X1_train,X2_train)
      Y_train <- factor(c(rep(1,N1),rep(-1,N2)))
      df_train <- list(X = X_train, class = Y_train)    
      
      #test
      X1_test <- matrix(0,N1,p)
      X2_test <- matrix(0,N2,p)
      for (j in 1:p){
        X1_test[,j] <- runif(N1,0,1)   #class 1
        X2_test[,j] <- runif(N2,0.3,1.3)  #class 2
      }
      X_test <- rbind(X1_test,X2_test)
      Y_test <- factor(c(rep(1,N1),rep(-1,N2)))
      df_test <- list(X = X_test,class = Y_test)
    }
    if (case == "normMIX333"){
      corr1 <- 0.5; corr2 <- 0.5; corr3 <- 0.5
      SigmaX <- matrix(rep(0,p^2),p,p)
      for (i in 1:p){
        for(j in 1:p){
          SigmaX[i,j] <- corr1^(abs(i-j))
        }
      }
      Sigma1 <- SigmaX
      Sigma2 <- SigmaX
      Sigma3 <- SigmaX
      
      #train
      X11_train <- 1/3*mvrnorm(0.5*N1,c(rep(0,p/2),rep(3,p/2)),Sigma1)
      X12_train <- 1/3*mvrnorm(0.25*N1,c(rep(-3,p/2),rep(5,p/2)),Sigma2)
      X13_train <- 1/3*mvrnorm(0.25*N1,c(rep(-3,p/2),rep(-3,p/2)),Sigma3)
      X21_train <- 1/3*mvrnorm(0.5*N2,c(rep(0,p/2),rep(-3,p/2)),Sigma1)
      X22_train <- 1/3*mvrnorm(0.25*N2,c(rep(3,p/2),rep(-5,p/2)),Sigma2)
      X23_train <- 1/3*mvrnorm(0.25*N2,c(rep(3,p/2),rep(5,p/2)),Sigma3)
      X_train <- rbind(X11_train,X12_train,X13_train,X21_train,X22_train,X23_train)
      Y_train <- factor(c(rep(1,N1),rep(-1,N2)))
      df_train <- list(X = X_train, class = Y_train)     
      #test
      X11_test <- 1/3*mvrnorm(0.5*N1,c(rep(0,p/2),rep(3,p/2)),Sigma1)
      X12_test <- 1/3*mvrnorm(0.25*N1,c(rep(-3,p/2),rep(5,p/2)),Sigma2)
      X13_test <- 1/3*mvrnorm(0.25*N1,c(rep(-3,p/2),rep(-3,p/2)),Sigma3)
      X21_test <- 1/3*mvrnorm(0.5*N2,c(rep(0,p/2),rep(-3,p/2)),Sigma1)
      X22_test <- 1/3*mvrnorm(0.25*N2,c(rep(3,p/2),rep(-5,p/2)),Sigma2)
      X23_test <- 1/3*mvrnorm(0.25*N2,c(rep(3,p/2),rep(5,p/2)),Sigma3)
      X_test <- rbind(X11_test,X12_test,X13_test,X21_test,X22_test,X23_test)
      Y_test <- factor(c(rep(1,N1),rep(-1,N2)))
      df_test <- list(X = X_test, class = Y_test) 
      
    }
    if (case == "T3"){
      # #train
      corr <- 0.5   
      SigmaX <- diag(1,p)
      cc <- 0.75
      X1_train <- rmt(N1, rep(-cc, p), 1*SigmaX, 3)/10
      X2_train <- rmt(N2, rep(cc, p), 1*SigmaX, 3)/10   
      X_train <- rbind(X1_train,X2_train)
      Y_train <- factor(c(rep(1,N1),rep(-1,N2)))
      df_train <- list(X = X_train, class = Y_train)    
      
      #test
      X1_test <- rmt(N1, rep(-cc, p), 1*SigmaX, 3)/10
      X2_test <- rmt(N2, rep(cc, p), 1*SigmaX, 3)/10     
      X_test <- rbind(X1_test,X2_test)
      Y_test <- factor(c(rep(1,N1),rep(-1,N2)))
      df_test <- list(X = X_test, class = Y_test)
    }
    if (case == "T3MIX222"){
      corr <- 0.5
      SigmaX = diag(rep(1,p))
      Sigma1 <- SigmaX
      Sigma2 <- SigmaX
      df <- 3
      cc <- 0.3
      aa <- 0.4
      bb <- 1 
      #train
      X11_train <- 1*rmt(cc*N1,c(rep(2,p/2),rep(2,p/2)),Sigma1,df)/bb
      X12_train <- 1*rmt((1-cc)*N1,c(rep(-3,p/2),rep(-3,p/2)),Sigma2,df)/bb
      X21_train <- 1*rmt(aa*N2,c(rep(-1,p/2),rep(-1,p/2)),Sigma1,df)/bb
      X22_train <- 1*rmt((1-aa)*N2,c(rep(8,p/2),rep(8,p/2)),Sigma2,df)/bb
      X_train <- rbind(X11_train,X12_train,X21_train,X22_train)
      Y_train <- factor(c(rep(1,N1),rep(-1,N2)))
      df_train <- list(X = X_train, class = Y_train)     
      #test
      X11_test <- 1*rmt(cc*N1,c(rep(2,p/2),rep(2,p/2)),Sigma1,df)/bb
      X12_test <- 1*rmt((1-cc)*N1,c(rep(-3,p/2),rep(-3,p/2)),Sigma2,df)/bb
      X21_test <- 1*rmt(aa*N2,c(rep(-1,p/2),rep(-1,p/2)),Sigma1,df)/bb
      X22_test <- 1*rmt((1-aa)*N2,c(rep(8,p/2),rep(8,p/2)),Sigma2,df)/bb
      X_test <- rbind(X11_test,X12_test,X21_test,X22_test)
      Y_test <- factor(c(rep(1,N1),rep(-1,N2)))
      df_test <- list(X = X_test, class = Y_test) 
      
    }
  }
  return (list(df_train = df_train,df_test = df_test))
  
}

#tuning by GACV
func_gacv <- function(dat_train, range, scale,kernel,weight){
  
  X <- dat_train$X
  XC <- dat_train$XC
  Y <- dat_train$class
  Y_num <- dat_train$Y_num
  d_train <- data.frame(X = X, class = Y)
  
  N <- nrow(X)
  p <- ncol(X)
  
  ff <- function(cost){
    fit <- wsvm(class~., data = d_train, weight = weight, kernel= kernel, cost= cost, scale = scale)
    beta <- (coef(fit)) #include intercept 
    
    alpha <- rep(0,N)
    temp_alpha <- abs(fit$coefs)  #support vector index
    alpha[fit$index] <- temp_alpha
    
    
    temp <- Y_num*(XC%*%beta)
    index1 <- which(temp<(-1))
    part1 <- 2*(cost)*sum(alpha[index1]*(diag(X[index1,]%*%t(X[index1,]))))
    index2 <- which(temp>=-1&temp<=1)
    part2 <- cost*sum(alpha[index2]*(diag(X[index2,]%*%t(X[index2,]))))
    
    Dcost <- (part1+part2)/N
    
    index <- which(temp<=1)
    loss <- (sum((rep(1,length(index))-temp[index])))/N
    
    gacv <- loss+Dcost
    
    return(gacv)
  }
  temperror <- sapply(range, ff)
  cost <- range[which.min(temperror)]
  return(list(cost = cost, temperror =temperror ))
}

#standard binary weighted SVM  
SVM_full <- function(dat_train,dat_test,range, scale,nfold, kernel, weight){
  
  # training data
  X <- dat_train$X
  XC <- dat_train$XC
  Y <- dat_train$class
  Y_num <- dat_train$Y_num
  d_train <- data.frame(X=X, class=Y)
  N <- nrow(X)
  p <- ncol(X)
  
  #testing data
  X_test <- dat_test$X
  XC_test <- dat_test$XC_test
  Y_test <- dat_test$class
  d_test <- data.frame(X = X_test, class = Y_test)
  
  #tuning by CV or GACV
  tic_tune <- proc.time()
  ####tune <- func_gacv(dat_train, range, scale,kernel, weight)     #GACV
  tune <- func_liblineaR(dat_train, range, scale, kernel)   #LiblineaR, CV
  cost <- tune$cost
  temperror <- tune$temperror
  time_tune <- (proc.time()-tic_tune)[3]
  
  #training
  tic_train <- proc.time()
  fit_SVM <- wsvm(class~., data = d_train, weight = weight, kernel= kernel, cost= cost, scale = scale)  #weight,WeightSVM
  time_train <- (proc.time()-tic_train)[3]        
  
  
  beta <- (coef(fit_SVM)) #include intercept 
  
  cost <- fit_SVM$cost
  datasv <- fit_SVM$SV
  nSV <-fit_SVM$tot.nSV  
  id_SVM <- as.numeric(row.names(datasv))
  datasv_SVM <- data.frame( x= datasv[,1:(p-1)], z= datasv[,p], method = rep("SVM", nSV))  
  
  pred_svm <- predict(fit_SVM, newdata = dat_test$X) #SVM classifier
  cft <- table(pred_svm, dat_test$class)
  TP <- cft[1,1]
  TN <- cft[2,2]
  FP <- cft[2,1]
  FN <- cft[1,2]
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  miserror <- (FP+FN)/(TP+TN+FP+FN)
  
  return(list(accuracy = accuracy, miserror = miserror,temperror = temperror,
              beta = beta, cost = cost, nSV = nSV, id = id_SVM,datasv = datasv_SVM,
              time_train =time_train, time_tune = time_tune))
}

#SVM for uniform sampling
SVM <- function(dat_train,dat_test,range, scale,nfold, kernel, weight){
  # training data
  X <- dat_train$X
  XC <- dat_train$XC
  Y <- dat_train$class
  Y_num <- dat_train$Y_num
  d_train <- data.frame(X=X, class=Y)
  N <- nrow(X)
  p <- ncol(X)
  
  #testing data
  X_test <- dat_test$X
  XC_test <- dat_test$XC_test
  Y_test <- dat_test$class
  d_test <- data.frame(X = X_test, class = Y_test)
  
  
  #tuning by CV or GACV
  tic_tune <- proc.time()
  tune <- func_gacv(dat_train, range, scale,kernel, weight)     #GACV
  cost <- tune$cost
  temperror <- tune$temperror
  time_tune <- (proc.time()-tic_tune)[3]
  
  #training
  tic_train <- proc.time()
  fit_SVM <- wsvm(class~., data = d_train, weight = weight, kernel= kernel, cost= cost, scale = scale)  #weight,WeightSVM
  time_train <- (proc.time()-tic_train)[3]        
  
  
  beta <- (coef(fit_SVM)) #include intercept 
  
  cost <- fit_SVM$cost
  datasv <- fit_SVM$SV
  nSV <-fit_SVM$tot.nSV  
  id_SVM <- as.numeric(row.names(datasv))
  datasv_SVM <- data.frame( x= datasv[,1:(p-1)], z= datasv[,p], method = rep("SVM", nSV))  
  
  pred_svm <- predict(fit_SVM, newdata = dat_test$X) #SVM classifier
  cft <- table(pred_svm, dat_test$class)
  TP <- cft[1,1]
  TN <- cft[2,2]
  FP <- cft[2,1]
  FN <- cft[1,2]
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  miserror <- (FP+FN)/(TP+TN+FP+FN)
  
  return(list(accuracy = accuracy, miserror = miserror,temperror = temperror,
              beta = beta, cost = cost, nSV = nSV, id = id_SVM,datasv = datasv_SVM,
              time_train =time_train, time_tune = time_tune))
}


#Smooth function, Wang et al 2019
Ks <- function(u){
  if (u<=-1)     {Ks = 0}
  if (abs(u)<1)  {Ks = 0.5+15/16*(u-2/3*u^3+1/5*u^5)} #{Ks <- 0.75*(1-u^2)}
  if (u >=1)     {Ks = 1}
  
  return(Ks)
}

Ks_derivative <- function(u){
  if (u<=-1)     {Ks = 0}
  if (abs(u)<1)  {Ks = 15/16*(1-2*u^2+u^4)} # {Ks <- -1.5*u}#
  if (u >=1)     {Ks = 0}
  
  return(Ks)
}

matpower <- function(a,b){
  a <- round((a+t(a))/2,7)
  tem <- eigen(a)
  return(tem$vectors%*%diag((tem$values)^b)%*%t(tem$vectors))
}

##two step SVM
#OS,max(pi,delta_N), using indicator function approximation
twostep_SVM_OS <- function(dat_train, dat_test,range, scale, nfold, kernel, replace, n0, n) {
  
  # training data
  X <- dat_train$X
  XC <- dat_train$XC
  Y <- dat_train$class
  Y_num <- dat_train$Y_num
  d_train <- data.frame(X=X, class=Y)
  N <- nrow(X)
  p <- ncol(X)
  
  #testing data
  X_test <- dat_test$X
  XC_test <- dat_test$XC
  Y_test <- dat_test$class
  d_test <- data.frame(X = X_test, class = Y_test)
  
  ############pilot study: stage1
  N1 <- length(which(Y==1))                        #number of Y=1
  N0 <- N - N1                                     #number of Y=-1
  PI_pilot <- rep(1/(2*N0), N)
  PI_pilot[Y==1] <- 1/(2*N1)
  #PI_pilot <- rep(1/N,N)
  idx_pilot <- sort(sample(1:N, n0, replace = replace, PI_pilot)) #step 1 subsample id without replacement
  pinv_pilot <- 1/(PI_pilot[idx_pilot]) #1/pi*, maintain the order in full data
  
  X_pilot <- X[idx_pilot,]
  XC_pilot <- XC[idx_pilot,]
  Y_pilot <- Y[idx_pilot]
  Y_num_pilot <- Y_num[idx_pilot]
  dat_pilot <- list(X = X_pilot, XC = XC_pilot, class = Y_pilot,Y_num = Y_num_pilot)  #tuning SVM needs Y num
  d_pilot <- data.frame(X = X_pilot,class = Y_pilot)  #fitting SVM needs Y factor
  
  
  #tuning by CV or GACV
  tic_pilot_tune <- proc.time()
  cost_pilot_tune <- func_gacv(dat_pilot, range,scale, kernel, weight=pinv_pilot/N)      #GACV
  cost_pilot <- cost_pilot_tune$cost
  temperror_pilot <- cost_pilot_tune$temperror
  time_pilot_tune <- (proc.time()-tic_pilot_tune)[3]
  
  
  tic_pilot_train <- proc.time()
  fit_pilot <- wsvm(class~., data = d_pilot, weight = pinv_pilot/N, kernel=kernel, cost= cost_pilot, scale = scale) #weight for each data point, WeightedSVM
  time_pilot_train <- (proc.time()-tic_pilot_train)[3]
  
  
  beta_pilot <- (coef(fit_pilot))
  xi <- 0.01/N
  #################subsample: stage2
  
  #method=="mMSE"
  {
    #######subsample
    tic_PI_mMSE <- proc.time()
    
    
    #first derivative
    temp_full <- 1-Y_num*(XC%*%beta_pilot)
    S <- rep(0,N)
    S[which(temp_full>=0)] <- 1         #indicator function
    S[which(temp_full<0)] <- 0          #loss equals to 0; i.e., the points are away from hyperplane
    
    #second derivative,Koo(2008),nonparametric estimate based on pilot data
    temp_pilot <- 1 - Y_num_pilot*(XC_pilot%*%beta_pilot)   # kernel estimation on pilot
    h_pilot <- bw.nrd0(XC_pilot)  #Sliverman rule of thumb
    #h_pilot <- bw.SJ(XC_pilot)    #Sheather and Jones
    #h_pilot <- bw.bcv(XC_pilot, nb =1000, lower = 0.001,upper =1) #Biased cross-validation (BCV)
    #h_pilot <- bw.ucv(XC_pilot, nb =1000, lower = 0.001,upper =1) #unbiased cross-validation (BCV)
    #h_pilot <- bw.nrd(XC_pilot)    #Scott rule of thumb
    
    st <- c(0.75*(1-(temp_pilot/h_pilot)^2)/h_pilot)
    KI <- diag(st)
    KI[KI<0] <- 0
    H <- t(XC_pilot)%*%KI%*%(XC_pilot*pinv_pilot)/(N*n0)
    
    #subsampling probability
    W_prop <- matpower(H,-1)
    PI_mMSE <- sqrt(S*rowSums((XC%*%W_prop)^2))
    PI_mMSE <- PI_mMSE/sum(PI_mMSE)
    
    
    time_PI_mMSE <- (proc.time()-tic_PI_mMSE)[3]
    
    idx_mMSE <- sample(1:N, n, replace = replace, PI_mMSE)
    
    PI_mMSE_sub <- c(PI_mMSE[idx_mMSE],1/pinv_pilot)
    idx_mMSE <- c(idx_mMSE,idx_pilot)                   #n0+n
    
    PI_mMSE_sub <- PI_mMSE_sub[order(idx_mMSE)]  #return the location of subsample in full data
    idx_mMSE <- sort(idx_mMSE)
    nSV_mMSE <- length(idx_mMSE)
    pinv_mMSE_sub <- c(1/PI_mMSE_sub)
    
    #subsample by PI
    X_mMSE_sub <- X[c(idx_mMSE),]
    XC_mMSE_sub <- XC[c(idx_mMSE),]
    Y_mMSE_sub <- Y[c(idx_mMSE)]
    Y_num_mMSE_sub <- Y_num[c(idx_mMSE)]
    dat_mMSE <- list(X = X_mMSE_sub, class = Y_mMSE_sub, XC = XC_mMSE_sub,Y_num=Y_num_mMSE_sub)
    d_mMSE <- data.frame(X = X_mMSE_sub, class = Y_mMSE_sub)
    subdata_mMSE <- data.frame(x=X[idx_mMSE,1:(p-1)], z= X[idx_mMSE,p],method = rep('mMSE', length(idx_mMSE))) #subdata
    
    tic_mMSE_tune <- proc.time()
    cost_mMSE_tune <- func_gacv(dat_mMSE,range,scale,kernel,weight=pinv_mMSE_sub/N)  #GACV
    
    
    cost_mMSE <- cost_mMSE_tune$cost
    temperror_mMSE <- cost_mMSE_tune$temperror
    time_mMSE_tune <- (proc.time()-tic_mMSE_tune)[3]
    
    ##train SVM on subsample
    tic_mMSE_train <- proc.time()
    fit_mMSE <- wsvm(class~., data = d_mMSE, weight = pinv_mMSE_sub/N, kernel=kernel,cost= cost_mMSE, scale = scale) #WeightedSVM
    time_mMSE_train <- (proc.time()-tic_mMSE_train)[3]
    
    beta_mMSE <- (coef(fit_mMSE))
    
    cost_mMSE <- fit_mMSE$cost
    nSV_mMSE <- fit_mMSE$tot.nSV
    datasv_mMSE <- fit_mMSE$SV      #support vector on subdata
    id_mMSE <- as.numeric(row.names(datasv_mMSE))
    
    ##preidct on testing data
    pred_svm_mMSE <- predict(fit_mMSE, newdata = dat_test$X) #SVM classifier
    cft_mMSE <- table(pred_svm_mMSE, dat_test$class)
    TP_mMSE <- cft_mMSE[1,1]
    TN_mMSE <- cft_mMSE[2,2]
    FP_mMSE <- cft_mMSE[2,1]
    FN_mMSE <- cft_mMSE[1,2]
    accuracy_mMSE <- (TP_mMSE+TN_mMSE)/(TP_mMSE+TN_mMSE+FP_mMSE+FN_mMSE)
    miserror_mMSE <-  (FP_mMSE+FN_mMSE)/(TP_mMSE+TN_mMSE+FP_mMSE+FN_mMSE)
  }
  
  #method=="mVc"
  {
    #######subsample
    tic_PI_mVc <- proc.time()
    
    #first derivative
    temp_full <- 1-Y_num*(XC%*%beta_pilot)
    S <- rep(0,N)
    S[which(temp_full>=0)] <- 1         #indicator function
    S[which(temp_full<0)] <- 0          #loss equals to 0; i.e., the points are away from hyperplane
    
    
    #subsampling probability   
    PI_mVc <- sqrt(S*rowSums(XC^2))
    PI_mVc <- PI_mVc/sum(PI_mVc)
    time_PI_mVc <- (proc.time()-tic_PI_mVc)[3]
    
    idx_mVc <- sample(1:N, n, replace = replace, PI_mVc)
    
    PI_mVc_sub <- c(PI_mVc[idx_mVc],1/pinv_pilot)
    idx_mVc <- c(idx_mVc,idx_pilot)                   #n0+n
    
    PI_mVc_sub <- PI_mVc_sub[order(idx_mVc)]      #return the location of subsample in full data
    idx_mVc <- sort(idx_mVc)
    nSV_mVc <- length(idx_mVc)
    pinv_mVc_sub <- c(1/PI_mVc_sub)  #ensure pinv and data point one-to-one correspondence
    
    #subsample by PI
    X_mVc_sub <- X[c(idx_mVc),]
    XC_mVc_sub <- XC[c(idx_mVc),]
    Y_mVc_sub <- Y[c(idx_mVc)]
    Y_num_mVc_sub <- Y_num[c(idx_mVc)]
    dat_mVc <- list(X = X_mVc_sub, class = Y_mVc_sub,XC = XC_mVc_sub,Y_num = Y_num_mVc_sub)
    d_mVc <- data.frame(X = X_mVc_sub, class = Y_mVc_sub)
    subdata_mVc <- data.frame( x=X[idx_mVc,1:(p-1)], z= X[idx_mVc,p],method = rep('mVc', length(idx_mVc)))
    
    
    tic_mVc_tune <- proc.time()
    cost_mVc_tune <- func_gacv(dat_mVc,range,scale,kernel,weight=pinv_mVc_sub/N)   #GACV
    
    
    cost_mVc <- cost_mVc_tune$cost
    temperror_mVc <- cost_mVc_tune$temperror
    time_mVc_tune <- (proc.time()-tic_mVc_tune)[3]
    
    ##train SVM on subsample
    tic_mVc_train <- proc.time()
    fit_mVc <- wsvm(class~., data = d_mVc, weight = pinv_mVc_sub/N, kernel=kernel,cost= cost_mVc, scale = scale)
    time_mVc_train <- (proc.time()-tic_mVc_train)[3]
    beta_mVc <- (coef(fit_mVc))
    
    cost_mVc <- fit_mVc$cost
    nSV_mVc <- fit_mVc$tot.nSV
    datasv_mVc <- fit_mVc$SV
    id_mVc <- as.numeric(row.names(datasv_mVc))
    
    pred_svm_mVc <- predict(fit_mVc, newdata = dat_test$X) #SVM classifier
    cft_mVc <- table(pred_svm_mVc, dat_test$class)
    TP_mVc <- cft_mVc[1,1]
    TN_mVc <- cft_mVc[2,2]
    FP_mVc <- cft_mVc[2,1]
    FN_mVc <- cft_mVc[1,2]
    accuracy_mVc <- (TP_mVc+TN_mVc)/(TP_mVc+TN_mVc+FP_mVc+FN_mVc)
    miserror_mVc <-  (FP_mVc+FN_mVc)/(TP_mVc+TN_mVc+FP_mVc+FN_mVc)
    
  }
  return(list(accuracy_mMSE = accuracy_mMSE, miserror_mMSE = miserror_mMSE,
              beta_mMSE = beta_mMSE, cost_mMSE = cost_mMSE, nSV_mMSE = nSV_mMSE,
              datasv_mMSE = datasv_mMSE, subdata_mMSE = subdata_mMSE,
              time_PI_mMSE = time_PI_mMSE, time_mMSE_tune = time_mMSE_tune, time_mMSE_train = time_mMSE_train,
              accuracy_mVc = accuracy_mVc, miserror_mVc = miserror_mVc,
              beta_mVc = beta_mVc, cost_mVc = cost_mVc, nSV_mVc = nSV_mVc,
              datasv_mVc = datasv_mVc, subdata_mVc = subdata_mVc,
              time_PI_mVc = time_PI_mVc, time_mVc_tune = time_mVc_tune, time_mVc_train = time_mVc_train,
              beta_pilot = beta_pilot, time_pilot_tune = time_pilot_tune, time_pilot_train = time_pilot_train))
}

#MSE between two hyper-planes without intercept
dist_MSE <- function(a,b){
  
  ff <- function(x){
    w <- x[-1]
    #w <- x
    norm <- as.numeric((sqrt((t(w)%*%w))))
    y <- x/norm
    return(y)
  }       # standardization for different beta 
  a <- (ff(a))
  b <- (ff(b))
  
  d1 = sum((a-b)^2)
  d2 = sum((a-(-b))^2)
  d = min(d1,d2)
  return(d)
}

#fix full data
OptimalSamp <- function(result_SVM,dat_train,dat_test,N,delta,p,n0,n,range,kernel,replace,scale,case,seedNo){
  #set.seed(2020)
  set.seed(123+1000*seedNo)
  
  acc_SVM <- result_SVM$acc_SVM
  time_SVM <- result_SVM$time_SVM
  time_SVM_train <- result_SVM$time_SVM_train
  cost_SVM <- result_SVM$cost_SVM
  beta_SVM <- result_SVM$beta_SVM
  time_SVM_tune <- result_SVM$time_SVM_tun
  
  #Sub SVM: uniform sampling
  if(n<N){
    # # #random sampling
    PI_uni <- rep(1/N,N)
    id_uni <- sort(sample(1:N,n0+n,replace = replace, PI_uni))
  }
  if(n==N){id_uni <- 1:N}
  
  weight_uni <- 1/(N*rep(1/N,length(id_uni))) #random sampling pi=1/full data size, unbaised, weight=1/pi
  
  X <- dat_train$X
  XC <- dat_train$XC
  Y <- dat_train$class
  Y_num <- dat_train$Y_num
  dat_train_uni <- list(X=X[id_uni,],XC=XC[id_uni,],class=Y[id_uni],Y_num=as.numeric(as.character(Y[id_uni])))
  dat_test_uni <- dat_test
  fit_SVM_uni <- SVM(dat_train_uni,dat_test_uni,range,scale,nfold,kernel,weight_uni)  #model SVM: construct model on trianing data , tuning on tuning data
  acc_SVM_uni <- fit_SVM_uni$accuracy                    #accuracy
  error_SVM_uni <- fit_SVM_uni$miserror                  #misclassification error rate in test data
  nSV_SVM_uni <- fit_SVM_uni$nSV                         #number of support vectors
  id_SVM_uni <- fit_SVM_uni$id                       #support vector id
  datasv_SVM_uni <- fit_SVM_uni$datasv                #support vector subdata
  beta_SVM_uni <- fit_SVM_uni$beta
  MSE_SVM_uni <- dist_MSE(beta_SVM,beta_SVM_uni)
  cost_SVM_uni <- fit_SVM_uni$cost
  temperror_SVM_uni <- fit_SVM_uni$temperror
  time_SVM_uni_train <- fit_SVM_uni$time_train
  time_SVM_uni_tune <- fit_SVM_uni$time_tune
  time_SVM_uni <- time_SVM_uni_train + time_SVM_uni_tune
  subdata_SVM_uni <- data.frame(x= (dat_train$X)[id_uni,p-1], z= (dat_train$X)[id_uni,p], method = rep("SVM", length(id_uni))) #toy example
  
  #optimal subsampling in SVM,OS
  fit <- twostep_SVM_OS(dat_train,dat_test,range, scale, nfold, kernel, replace, n0, n)
  #mMSE,A-optimal
  acc_mMSE <- fit$accuracy_mMSE
  error_mMSE <- fit$miserror_mMSE
  nSV_mMSE <- fit$nSV_mMSE
  id_mMSE <- fit$id_mMSE
  subdata_mMSE <- fit$subdata_mMSE
  beta_mMSE <- fit$beta_mMSE
  MSE_mMSE <- dist_MSE(beta_SVM,beta_mMSE)
  cost_mMSE <- fit$cost_mMSE
  time_pilot <- fit$time_pilot_train + fit$time_pilot_tune
  time_PI_mMSE <- fit$time_PI_mMSE
  time_fit_mMSE <- fit$time_mMSE_train + fit$time_mMSE_tune
  time_mMSE <- time_pilot + time_PI_mMSE + time_fit_mMSE
  time_mMSE_train <- fit$time_pilot_train + fit$time_PI_mMSE +fit$time_mMSE_train
  time_mMSE_tune <- fit$time_pilot_tune + fit$time_mMSE_tune
  #mVc,L-optimal
  acc_mVc <- fit$accuracy_mVc
  error_mVc <- fit$miserror_mVc
  nSV_mVc <- fit$nSV_mVc
  id_mVc <- fit$id_mVc
  subdata_mVc <- fit$subdata_mVc
  beta_mVc <- fit$beta_mVc
  MSE_mVc <- dist_MSE(beta_SVM,beta_mVc)
  cost_mVc <- fit$cost_mVc
  time_pilot <- fit$time_pilot_train + fit$time_pilot_tune
  time_PI_mVc <- fit$time_PI_mVc
  time_fit_mVc <- fit$time_mVc_train + fit$time_mVc_tune
  time_mVc <- time_pilot + time_PI_mVc + time_fit_mVc
  time_mVc_train <- fit$time_pilot_train + fit$time_PI_mVc +fit$time_mVc_train
  time_mVc_tune <- fit$time_pilot_tune + fit$time_mVc_tune
  
  return(c(acc_mMSE = acc_mMSE, acc_mVc = acc_mVc, acc_SVM = acc_SVM, acc_SVM_uni = acc_SVM_uni,
           time_mMSE = time_mMSE, time_mVc = time_mVc, time_SVM = time_SVM, time_SVM_uni = time_SVM_uni,
           time_mMSE_train = time_mMSE_train, time_mVc_train = time_mVc_train, time_SVM_train = time_SVM_train, time_SVM_uni_train = time_SVM_uni_train,
           time_mMSE_tune = time_mMSE_tune, time_mVc_tune = time_mVc_tune, time_SVM_tune = time_SVM_tune, time_SVM_uni_tune = time_SVM_uni_tune,
           cost_mMSE = cost_mMSE, cost_mVc = cost_mVc, cost_SVM = cost_SVM, cost_SVM_uni = cost_SVM_uni,
           MSE_mMSE = MSE_mMSE, MSE_mVc = MSE_mVc, MSE_SVM_uni = MSE_SVM_uni))
}

#LiblineaR
func_liblineaR <- function(dat_train, range, scale, kernel){
  
  X <- dat_train$X 
  XC <- dat_train$XC
  Y <- dat_train$class
  Y_num <- dat_train$Y_num
  d_train <- data.frame(X = X, class = Y)
  
  N <- nrow(X)
  p <- ncol(X)
  
  ff <- function(cost){
    acc <- LiblineaR(data= scale(X,center=TRUE,scale=TRUE),target = Y, cost= cost, cross = 5, verbose=FALSE) #LiblineaR, no sample weight
    return(acc)
  }
  acc <- sapply(range, ff)
  cost <- range[which.max(acc)]
  temperror <- 1 - acc
  
  return(list(cost = cost, temperror =temperror ))
}

SVM_LiblineaR <- function(dat_train,dat_test,range, scale,nfold, kernel, weight){
  
  # training data
  X <- dat_train$X
  XC <- dat_train$XC
  Y <- dat_train$class
  Y_num <- dat_train$Y_num
  d_train <- data.frame(X=X, class=Y)
  N <- nrow(X)
  p <- ncol(X)
  
  #testing data
  X_test <- dat_test$X
  XC_test <- dat_test$XC_test
  Y_test <- dat_test$class
  d_test <- data.frame(X = X_test, class = Y_test)
  
  
  #tuning by CV or GACV
  tic_tune <- proc.time()
  tune <- func_liblineaR(dat_train, range, scale, kernel)   #LiblineaR, CV
  cost <- tune$cost
  temperror <- tune$temperror
  time_tune <- (proc.time()-tic_tune)[3]
  
  #training
  tic_train <- proc.time()
  fit_SVM <- LiblineaR(data=scale(X,center=TRUE,scale=TRUE),target = Y, cost= cost, type = 1)
  time_train <- (proc.time()-tic_train)[3]        
  
  
  beta <- c(fit_SVM$W[,p+1], fit_SVM$W[,1:p])
  pred_svm <- predict(fit_SVM, scale(dat_test$X),center = TRUE, scale = TRUE)$predictions
  cft <- table(pred_svm, dat_test$class)
  TP <- cft[1,1]
  TN <- cft[2,2]
  FP <- cft[2,1]
  FN <- cft[1,2]
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  miserror <- (FP+FN)/(TP+TN+FP+FN)
  
  return(list(accuracy = accuracy, miserror = miserror,temperror = temperror,
              beta = beta, cost = cost,time_train =time_train, time_tune = time_tune))
}