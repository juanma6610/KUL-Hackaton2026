## XGBoost + MCM model


## HLR Model
==== mean absolute error ====

	Welch Two Sample t-test

data:  abs(data$p - data$pp) and abs(data$p - mean(data$p))
t = -154.6, df = 2444807, p-value < 2.2e-16
alternative hypothesis: true difference in means is less than 0
95 percent confidence interval:
        -Inf -0.04499495
sample estimates:
mean of x mean of y 
0.1292369 0.1747157 

==== area under the ROC curve ====
Setting levels: control = 0, case = 1
Setting direction: controls < cases

Call:
roc.formula(formula = round(p) ~ pp, data = data)

Data: pp in 138800 controls (round(p) 0) < 1146623 cases (round(p) 1).
Area under the curve: 0.5377

	Wilcoxon rank sum test with continuity correction

data:  round(data$p) and data$pp
W = 1.4739e+12, p-value < 2.2e-16
alternative hypothesis: true location shift is greater than 0

==== half-life correlation ====

	Spearman's rank correlation rho

data:  data$h and data$hh
S = 2.8278e+17, p-value < 2.2e-16
alternative hypothesis: true rho is not equal to 0
sample estimates:
      rho 
0.2011603 

## Leitner Model

==== mean absolute error ====

	Welch Two Sample t-test

data:  abs(data$p - data$pp) and abs(data$p - mean(data$p))
t = 171.95, df = 2138181, p-value = 1
alternative hypothesis: true difference in means is less than 0
95 percent confidence interval:
       -Inf 0.06052839
sample estimates:
mean of x mean of y 
0.2346706 0.1747157 

==== area under the ROC curve ====
Setting levels: control = 0, case = 1
Setting direction: controls < cases

Call:
roc.formula(formula = round(p) ~ pp, data = data)

Data: pp in 138800 controls (round(p) 0) < 1146623 cases (round(p) 1).
Area under the curve: 0.5422

	Wilcoxon rank sum test with continuity correction

data:  round(data$p) and data$pp
W = 1.4739e+12, p-value < 2.2e-16
alternative hypothesis: true location shift is greater than 0

==== half-life correlation ====

	Spearman's rank correlation rho

data:  data$h and data$hh
S = 3.8852e+17, p-value < 2.2e-16
alternative hypothesis: true rho is not equal to 0
sample estimates:
        rho 
-0.09754527 

## Pimsleur Model

==== mean absolute error ====

	Welch Two Sample t-test

data:  abs(data$p - data$pp) and abs(data$p - mean(data$p))
t = 627.38, df = 1823944, p-value = 1
alternative hypothesis: true difference in means is less than 0
95 percent confidence interval:
      -Inf 0.2711067
sample estimates:
mean of x mean of y 
0.4451135 0.1747157 

==== area under the ROC curve ====
Setting levels: control = 0, case = 1
Setting direction: controls < cases

Call:
roc.formula(formula = round(p) ~ pp, data = data)

Data: pp in 138800 controls (round(p) 0) < 1146623 cases (round(p) 1).
Area under the curve: 0.5103

	Wilcoxon rank sum test with continuity correction

data:  round(data$p) and data$pp
W = 1.4739e+12, p-value < 2.2e-16
alternative hypothesis: true location shift is greater than 0

==== half-life correlation ====

	Spearman's rank correlation rho

data:  data$h and data$hh
S = 4.0073e+17, p-value < 2.2e-16
alternative hypothesis: true rho is not equal to 0
sample estimates:
       rho 
-0.1320369 






