modelInfo <- list(label = "H2O.ai Random Forest",
                  library = c("h2o"),
                  type = c("Regression", "Classification"),
                  parameters = data.frame(parameter = c('mtries', 'ntrees', 'sample_rate', 'max_depth',
                                                        'nbins', 'col_sample_rate_per_tree', 
                                                        'balance_classes'),
                                          class = c(rep("numeric", 6), "logical"),
                                          label = c('# of Variables Sampled',  "Number of Trees", 
                                                    'Sample Rate','Max Tree Depth', 
                                                    'Number of Bins for Splitting', "Column Sample Rate per tree",
                                                    'Toggle Balance Classes')),
                  grid = function(x, y, len = NULL, search = "grid") {
                    if(search == "grid") {
                      out <- expand.grid(mtries = -1,
                                         ntrees = ifelse(len<=10,seq(50,by = 50,length.out = len),seq(50,to=500, length.out=len)),
                                         sample_rate = seq(0.1,1.0,length.out=len),
                                         max_depth = 20,
                                         nbins = 20,
                                         col_sample_rate_per_tree = c(0.8,1),
                                         balance_classes = c(FALSE,TRUE))
                    } else {
                      out <- data.frame(mtries = -1,
                                        ntrees = ifelse(len<=10,seq(50,by = 50,length.out = len),seq(50,to=500, length.out=len)),
                                        sample_rate = seq(0.1,1.0,length.out=len),
                                        max_depth = 20,
                                        nbins = 20,
                                        col_sample_rate_per_tree = c(0.8,1),
                                        balance_classes = c(FALSE,TRUE))
                    }
                    out <- out[!duplicated(out),]
                    out
                  },
                  fit = function(x, y, param, cleanH2O = TRUE, ...) {
                    # Initialize H2O
                    h2o.init(nthreads = parallel::detectCores(TRUE)-1)
                    if(cleanH2O){
                      message("Cleaning up all variables in H2O Environment...")
                      h2o.removeAll()
                    }
                    
                    # Create H2O data frames
                    train_h2o <- as.h2o(cbind(x,y),destination_frame = "caret_h2oRF_train")
                    x_h2o <- c(1:(ncol(train_h2o)-1))
                    y_h2o <- c(ncol(train_h2o))
                    
                    out <- h2o.randomForest(x_h2o,y_h2o,training_frame = train_h2o,model_id = "caret_h2oRF",
                                            mtries = param$mtries,sample_rate = param$sample_rate,
                                            col_sample_rate_per_tree = param$col_sample_rate_per_tree,
                                            ntrees = param$ntrees,max_depth = param$max_depth,nbins = param$nbins,
                                            balance_classes = param$balance_classes)
                    
                    out
                    
                    
                  },
                  predict = function(modelFit, newdata, submodels = NULL) {
                    newdata <- xgb.DMatrix(as.matrix(newdata))
                    out <- predict(modelFit, newdata)
                    if(modelFit$problemType == "Classification") {
                      if(length(modelFit$obsLevels) == 2) {
                        out <- ifelse(out >= .5, 
                                      modelFit$obsLevels[1], 
                                      modelFit$obsLevels[2])
                      } else {
                        out <- matrix(out, ncol = length(modelFit$obsLevels), byrow = TRUE)
                        out <- modelFit$obsLevels[apply(out, 1, which.max)]
                      }
                    }
                    
                    if(!is.null(submodels)) {
                      tmp <- vector(mode = "list", length = nrow(submodels) + 1)
                      tmp[[1]] <- out
                      for(j in seq(along = submodels$nrounds)) {
                        tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
                        if(modelFit$problemType == "Classification") {
                          if(length(modelFit$obsLevels) == 2) {
                            tmp_pred <- ifelse(tmp_pred >= .5, 
                                               modelFit$obsLevels[1], 
                                               modelFit$obsLevels[2])
                          } else {
                            tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), byrow = TRUE)
                            tmp_pred <- modelFit$obsLevels[apply(tmp_pred, 1, which.max)]
                          }
                        }
                        tmp[[j+1]]  <- tmp_pred
                      }
                      out <- tmp
                    }
                    out  
                  },
                  prob = function(modelFit, newdata, submodels = NULL) {
                    newdata <- xgb.DMatrix(as.matrix(newdata))
                    out <- predict(modelFit, newdata)
                    if(length(modelFit$obsLevels) == 2) {
                      out <- cbind(out, 1 - out)
                      colnames(out) <- modelFit$obsLevels
                    } else {
                      out <- matrix(out, ncol = length(modelFit$obsLevels), byrow = TRUE)
                      colnames(out) <- modelFit$obsLevels
                    }
                    out <- as.data.frame(out)
                    
                    if(!is.null(submodels)) {
                      tmp <- vector(mode = "list", length = nrow(submodels) + 1)
                      tmp[[1]] <- out
                      for(j in seq(along = submodels$nrounds)) {
                        tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
                        if(length(modelFit$obsLevels) == 2) {
                          tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
                          colnames(tmp_pred) <- modelFit$obsLevels
                        } else {
                          tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), byrow = TRUE)
                          colnames(tmp_pred) <- modelFit$obsLevels
                        }
                        tmp_pred <- as.data.frame(tmp_pred)
                        tmp[[j+1]]  <- tmp_pred
                      }
                      out <- tmp
                    }
                    out  
                  },
                  predictors = function(x, ...) {
                    imp <- xgb.importance(x$xNames, model = x)
                    x$xNames[x$xNames %in% imp$Feature]
                  },
                  varImp = function(object, numTrees = NULL, ...) {
                    imp <- xgb.importance(object$xNames, model = object)
                    imp <- as.data.frame(imp)[, 1:2]
                    rownames(imp) <- as.character(imp[,1])
                    imp <- imp[,2,drop = FALSE]
                    colnames(imp) <- "Overall"
                    imp   
                  },
                  levels = function(x) x$obsLevels,
                  tags = c("Tree-Based Model","Ensemble Model", "Bagging", "Implicit Feature Selection"),
                  sort = function(x) {
                    x[order(x$mtries, x$max_depth, x$ntrees, x$sample_rate, x$nbins, x$col_sample_rate_per_tree, x$balance_classes),] 
                  })
