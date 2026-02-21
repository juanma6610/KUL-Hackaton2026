# Copyright (c) 2016 Duolingo Inc. MIT Licence.

library('pROC')

sr_evaluate <- function(preds_file) {
    cat(paste('%%%%%%%%%%%%%%%%', preds_file, '%%%%%%%%%%%%%%%%\n'));
    data <- read.csv(preds_file, sep='\t');
    cat('==== mean absolute error ====\n')
    print(t.test(abs(data$p - data$pp), abs(data$p - mean(data$p)), alternative='l'));
    cat('==== area under the ROC curve ====\n')
    print(roc(round(p) ~ pp, data=data));
    print(wilcox.test(round(data$p), data$pp, alternative='g'));
    cat('==== half-life correlation ====\n')
    print(cor.test(data$h, data$hh, method='s'));
}



###################Their results
# ==== mean absolute error ====

# 	Welch Two Sample t-test

# data:  abs(data$p - data$pp) and abs(data$p - mean(data$p))
# t = -155.14, df = 2444853, p-value < 2.2e-16
# alternative hypothesis: true difference in means is less than 0
# 95 percent confidence interval:
#         -Inf -0.04515188
# sample estimates:
# mean of x mean of y 
# 0.1290800 0.1747157 

# ==== area under the ROC curve ====
# Setting levels: control = 0, case = 1
# Setting direction: controls < cases

# Call:
# roc.formula(formula = round(p) ~ pp, data = data)

# Data: pp in 138800 controls (round(p) 0) < 1146623 cases (round(p) 1).
# Area under the curve: 0.5377

# 	Wilcoxon rank sum test with continuity correction

# data:  round(data$p) and data$pp
# W = 1.4739e+12, p-value < 2.2e-16
# alternative hypothesis: true location shift is greater than 0

# ==== half-life correlation ====

# 	Spearman's rank correlation rho

# data:  data$h and data$hh
# S = 2.8281e+17, p-value < 2.2e-16
# alternative hypothesis: true rho is not equal to 0
# sample estimates:
#       rho 
# 0.2010585