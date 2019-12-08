##################################################
##################################################
##########
##########
rm(list=ls())

##################################################


##################################################
##################################################

## load libraries
library(binomialRF)
require(data.table)
require(parallel)
source('../../../fast_correlbinom.R')
set.seed(324)
setwd('~/Desktop/classes/fall 2019/data science for engineers/final_project/numerical studies/uci_ml_repo/code/')





readInData <- function(directory){
  
  Xvalid <- fread(paste('../data/',directory,'/',directory,'_valid.data',sep=''), data.table = F)
  Xtrain <- fread(paste('../data/',directory,'/',directory,'_train.data',sep=''), data.table = F)
  
  ytrain <- fread(paste('../data/',directory,'/',directory,'_train.labels',sep=''), data.table = F)
  yvalid <- fread(paste('../data/',directory,'/',directory,'_valid.labels',sep=''), data.table = F)
  
  #params <- fread(paste('../data/',directory,'/',directory,'.param',sep=''))
  dataType = paste(unique(sapply(1:ncol(Xvalid), function(i) class(Xvalid[,i]))),collapse = ' | ')
  
  params <- c(dim(Xvalid),dataType, directory, 'Classification')
  
  return(list(Xvalid=Xvalid,
              Xtrain=Xtrain,
              yvalid=yvalid,
              ytrain=ytrain,
              params=params))
}

directories <- dir('../data')
directories <- directories[!directories %in% c('otherFormat')]

data_list <- list()
for(direc in directories){
  data_list[[direc]] <- readInData(direc)
}

DataDict <- as.data.frame(do.call(rbind, lapply(directories, function(x) data_list[[x]]$params)))
colnames(DataDict) <- c('Instances','Attributes','AttributeType','Dataset', 'Task')
write.csv(DataDict, '../results/dataDict.csv', row.names=F)


analyzeData <- function(directory){
  
  dt <- data_list[[directory]]
  
  rho = 0.33
  X <- dt$Xtrain
  y <- as.factor(dt$ytrain$V1)
  X_test = dt$Xvalid
  y_test = as.factor(dt$yvalid$V1)
  
  ntrees=1000
  
  cbinom_dist <- fast_correlbinom(rho=rho, successprob = 1/ncol(X),trials = ntrees,model = 'kuk')
  
  run_replicates <- function(X,y, X_test,y_test, cbinom_dist = cbinom_dist){
    
    idx <- sample.int(nrow(X), size=nrow(X) * .60)
      
    ## CorBinomialRF
    combined.binomRF = binomialRF(X[idx,],y[idx] ,
                                  fdr.threshold=.05,
                                  fdr.method='BY',
                                  ntrees=ntrees,
                                  percent_features=.2,
                                  correlationAdjustment = T,
                                  keep.rf =FALSE,
                                  user_cbinom_dist=cbinom_dist,
                                  sampsize= nrow(X)*rho)
    
    cor.binomRF = combined.binomRF$cor.binomRF
    cor.binomRF_vars <- as.character(cor.binomRF$variable[ cor.binomRF$adjSignificance < .05])
    cor.binomRF_vars <- stringr::str_replace(string = cor.binomRF_vars,pattern = 'X','V')
    
    cor.rf <- randomForest::randomForest(X[idx,cor.binomRF_vars], y[idx])
    cor.rf_err <- mean(!predict(cor.rf, X_test) ==y_test)
    cor.rf_modelSize <- length(cor.binomRF_vars)
    
    
    ### binomialRF
    binomRF <- combined.binomRF$binomRF
    binomRF_vars <- as.character(binomRF$variable[binomRF$adjSignificance < 0.05])
    binomRF_vars <- stringr::str_replace(string = binomRF_vars,pattern = 'X','V')
    
    bin.rf <- randomForest::randomForest(X[,binomRF_vars], y)
    bin.rf_err <- mean(!predict(bin.rf, X_test) ==y_test)
    
    bin.rf_modelSize <- length(binomRF_vars)
    resultMatrix <- data.frame(ValidErr = c(bin.rf_err, cor.rf_err),
                               ModelSize= c( bin.rf_modelSize, cor.rf_modelSize),
                               Model = c('binomialRF','corBinomialRF'),
                               dimX = c(ncol(X),ncol(X)))
    
    return(resultMatrix)
    
  }
  
  results <- mclapply(1:100, function(zz) run_replicates(X,y, X_test,y_test, cbinom_dist = cbinom_dist))
  return(results) 
}

resultsArcene <- do.call(rbind, analyzeData(directory = 'arcene'))
resultsGisette <- do.call(rbind, analyzeData(directory = 'gisette'))
resultsMadelon <- do.call(rbind, analyzeData(directory = 'madelon'))

sum_results <- function(results100P_2000T){
  results100P_2000T2 = melt(results100P_2000T)
  results100P_2000T2 = data.table(results100P_2000T2)
  results100P_2000T2 = results100P_2000T2[, list(round(mean(value),2), round(sd(value),2)), by=list(Model, variable)]
  results100P_2000T2$value = paste(results100P_2000T2$V1,' (', results100P_2000T2$V2, ')', sep='')
  results100P_2000T2$value[results100P_2000T2$variable=='dimX'] <- results100P_2000T2$V1[results100P_2000T2$variable=='dimX']
  summary_results <- dcast(results100P_2000T2, Model ~ variable, value.var = 'value')
  
  return(summary_results)
}

arc = sum_results(resultsArcene)
gis = sum_results(resultsGisette)
mad = sum_results(resultsMadelon)

placeHolder = data.frame( Model = '', ValidErr = '',ModelSize= '', dimX = '')

final.mat <- rbind(arc, placeHolder,gis,placeHolder,mad)
print(final.mat)
write.csv(final.mat, file='../results/UCI_ml_results.csv', row.names = F)


