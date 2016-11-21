metricfun <- function(tp, fp, fn, tn)
{
 
  total = tp+fp+fn+tn
  acc = (tp+tn)/total
  randacc = ((tn+fp)*(tn+fn)+(fn+tp)*(fp+tp))/total^2
  kappa = (acc - randacc)/(1-randacc)
  sens = (tp)/(tp+fn)
  spec = tn/(fp + tn)
  prec = tp/(tp+fp)
  return(c(acc, sens, spec, prec, kappa))
}