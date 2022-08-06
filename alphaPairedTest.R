dir = "/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/Results/Music Therapy/BandPower/3.0/"  
TP9 = read.csv(paste(dir,"TP9.csv",sep=""), header=FALSE) 
AF7 = read.csv(paste(dir,"AF7.csv",sep=""), header=FALSE) 
AF8 = read.csv(paste(dir,"AF8.csv",sep=""), header=FALSE)
TP10 = read.csv(paste(dir,"TP10.csv",sep=""), header=FALSE)
meanChans = read.csv(paste(dir,"meanChannels.csv",sep=""), header=FALSE)
delta = read.csv(paste(dir,"delta.csv",sep=""), header=FALSE)
theta = read.csv(paste(dir,"theta.csv",sep=""), header=FALSE)
alpha = read.csv(paste(dir,"alpha.csv",sep=""), header=FALSE)
beta = read.csv(paste(dir,"beta.csv",sep=""), header=FALSE)
gamma = read.csv(paste(dir,"gamma.csv",sep=""), header=FALSE)

group_pre = rep('pre', 5) 
group_post = rep('post', 5)


delta_TP9_M11 = TP9[2:6,("V1")]
delta_AF7_M11 = AF7[2:6,("V1")]
delta_AF8_M11 = AF8[2:6,("V1")]
delta_TP10_M11 = TP10[2:6,("V1")]
delta_meanChans_M11 = meanChans[2:6,("V1")]
delta_TP9_C11 = TP9[7:11,("V1")]
delta_AF7_C11 = AF7[7:11,("V1")]
delta_AF8_C11 = AF8[7:11,("V1")]
delta_TP10_C11 = TP10[7:11,("V1")]
delta_meanChans_C11 = meanChans[7:11,("V1")]

delta_TP9_M12 = TP9[2:6,("V2")]
delta_AF7_M12 = AF7[2:6,("V2")]
delta_AF8_M12 = AF8[2:6,("V2")]
delta_TP10_M12 = TP10[2:6,("V2")]
delta_meanChans_M12 = meanChans[2:6,("V2")]
delta_TP9_C12 = TP9[7:11,("V2")]
delta_AF7_C12 = AF7[7:11,("V2")]
delta_AF8_C12 = AF8[7:11,("V2")]
delta_TP10_C12 = TP10[7:11,("V2")]
delta_meanChans_C12 = meanChans[7:11,("V2")]

#   Pre
delta_TP9_M11_Time = rbind(cbind(delta_TP9_M11, group_pre), cbind(delta_TP9_M12, group_post))
delta_AF7_M11_Time = rbind(cbind(delta_AF7_M11, group_pre), cbind(delta_AF7_M12, group_post))
delta_AF8_M11_Time = rbind(cbind(delta_AF8_M11, group_pre), cbind(delta_AF8_M12, group_post))
delta_TP10_M11_Time = rbind(cbind(delta_TP10_M11, group_pre), cbind(delta_TP10_M12, group_post))
delta_meanChans_M11_Time = rbind(cbind(delta_meanChans_M11, group_pre), cbind(delta_meanChans_M12, group_post))

delta_AF7_M11_pre = cbind(delta_AF7_M11, group_pre)
delta_AF8_M11_pre = cbind(delta_AF8_M11, group_pre)
delta_TP10_M11_pre = cbind(delta_TP10_M11, group_pre)
delta_meanChans_M11_pre = cbind(delta_meanChans_M11, group_pre)
delta_TP9_C11_pre = cbind(delta_TP9_C11, group_pre)
delta_AF7_C11_pre = cbind(delta_AF7_C11, group_pre)
delta_AF8_C11_pre = cbind(delta_AF8_C11, group_pre)
delta_TP10_C11_pre = cbind(delta_TP10_C11, group_pre)
delta_meanChans_C11_pre = cbind(delta_meanChans_C11, group_pre)

#   Post
delta_TP9_M12_post = cbind(delta_TP9_M12, group_post)
delta_AF7_M12_post = cbind(delta_AF7_M12, group_post)
delta_AF8_M12_post = cbind(delta_AF8_M12, group_post)
delta_TP10_M12_post = cbind(delta_TP10_M12, group_post)
delta_meanChans_M12_post = cbind(delta_meanChans_M12, group_post)
delta_TP9_C12_post = cbind(delta_TP9_C12, group_post)
delta_AF7_C12_post = cbind(delta_AF7_C12, group_post)
delta_AF8_C12_post = cbind(delta_AF8_C12, group_post)
delta_TP10_C12_post = cbind(delta_TP10_C12, group_post)
delta_meanChans_C12_post = cbind(delta_meanChans_C12, group_post)

#   Statistical Analysis: Compute T-test
#res <- t.test(delta_TP9_M11_pre,delta_TP9_M12_post,paired=TRUE)
