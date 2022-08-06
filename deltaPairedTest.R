dir = "/Users/joshuaighalo/Documents/BrainNet/Projects/Workspace/Results/Music Therapy/BandPower/3.0/"  
chansMean = read.csv(paste(dir,"meanChannels.csv",sep=""), header=FALSE)
delta = read.csv(paste(dir,"delta.csv",sep=""), header=FALSE)

group_pre = rep('pre', 5) 
group_post = rep('post', 5)

m11_chansMean = chansMean[2:6,("V1")]
c11_chansMean = chansMean[7:11,("V1")]
m12_chansMean = chansMean[2:6,("V2")]
c12_chansMean = chansMean[7:11,("V2")]
m21_chansMean = chansMean[2:6,("V3")]
c21_chansMean = chansMean[7:11,("V3")]
m22_chansMean = chansMean[2:6,("V4")]
c22_chansMean = chansMean[7:11,("V4")]

(ab) = (20)

m11_TP9 = delta[2:6,("V1")]
c11_TP9 = delta[7:11,("V1")]
m12_TP9 = delta[2:6,("V2")]
c12_TP9 = delta[7:11,("V2")]
m21_TP9 = delta[2:6,("V3")]
c21_TP9 = delta[7:11,("V3")]
m22_TP9 = delta[2:6,("V4")]
c22_TP9 = delta[7:11,("V4")]
m11_AF7 = delta[2:6,("V5")]
c11_AF7 = delta[7:11,("V5")]
m12_AF7 = delta[2:6,("V6")]
c12_AF7 = delta[7:11,("V6")]
m21_AF7 = delta[2:6,("V7")]
c21_AF7 = delta[7:11,("V7")]
m22_AF7 = delta[2:6,("V8")]
c22_AF7 = delta[7:11,("V8")]
m11_AF8 = delta[2:6,("V9")]
c11_AF8 = delta[7:11,("V9")]
m12_AF8 = delta[2:6,("V10")]
c12_AF8 = delta[7:11,("V10")]
m21_AF8 = delta[2:6,("V11")]
c21_AF8 = delta[7:11,("V11")]
m22_AF8 = delta[2:6,("V12")]
c22_AF8 = delta[7:11,("V12")]
m11_TP10 = delta[2:6,("V13")]
c11_TP10 = delta[7:11,("V13")]
m12_TP10 = delta[2:6,("V14")]
c12_TP10 = delta[7:11,("V14")]
m21_TP10 = delta[2:6,("V15")]
c21_TP10 = delta[7:11,("V15")]
m22_TP10 = delta[2:6,("V16")]
c22_TP10 = delta[7:11,("V16")]


