package LearningProb;

abstract class HMM_ABS {
  HMM hmm;
  String x; //OBSERVATION SEQUENCE
  
  public HMM_ABS(HMM hmm, String x){ this.hmm = hmm; this.x = x; }

  static double logLIMIT(double p, double q) {
    double max, diff;
    if (p > q) 
    {
      if (q == Double.NEGATIVE_INFINITY)
        return p;
      
      else {
        max = p; diff = q - p;
      } 
    } 
    else {
      if (p == Double.NEGATIVE_INFINITY)
        return q;
      else {
        max = q; diff = p - q;
      }
    }
    // Now diff <= 0 so Math.exp(diff) will not overflow
    return max + (diff < -37 ? 0 : Math.log(1 + Math.exp(diff)));
  }
}
class Forward extends HMM_ABS {
  double[][] f;                 // the matrix used to find the decoding
                                // f[i][k] = f_k(i) = log(P(x1..xi, pi_i=k))
  private int L; 

  public Forward(HMM hmm, String x) {
    super(hmm, x);
    
    L = x.length();
    f = new double[L+1][hmm.nstate];
    
    f[0][0] = 0;                // = log(1)
    for (int k=1; k<hmm.nstate; k++)
      f[0][k] = Double.NEGATIVE_INFINITY; // = log(0)
    for (int i=1; i<=L; i++)
      f[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
    
    for (int i=1; i<=L; i++)
      for (int ell=1; ell<hmm.nstate; ell++) {
        double sum = Double.NEGATIVE_INFINITY; // = log(0)
        for (int k=0; k<hmm.nstate; k++) 
          sum = logLIMIT(sum, f[i-1][k] + hmm.log_transMat[k][ell]);
        f[i][ell] = hmm.log_emiMat[ell][x.charAt(i-1)] + sum;
      }
  }
  double logprob() {
    double sum = Double.NEGATIVE_INFINITY; // = log(0)
    for (int k=0; k<hmm.nstate; k++) 
      sum = logLIMIT(sum, f[L][k]);
    return sum;
  }
}

class HMM {
	  // State names and state-to-state transition probabilities
	  int nstate;
	  String[] state;
	  double[][] log_transMat;

	  // Emission names and emission probabilities
	  int nEmi_symb;
	  String Emi_symb;
	  double[][] log_emiMat;

	  public HMM(String[] state, double[][] trans_mat, String Emi_symb, double[][] emi_mat) 
	  {
	    if (state.length != trans_mat.length)
	      throw new IllegalArgumentException("Error! Insufficent arguments");
	    
	    if (trans_mat.length != emi_mat.length)
	      throw new IllegalArgumentException("Error! Insufficent arguments");
	    
	    for (int i=0; i<trans_mat.length; i++) {
	      if (state.length != trans_mat[i].length)
	        throw new IllegalArgumentException("Error!transition mat is non-square");
	      
	      if (Emi_symb.length() != emi_mat[i].length)
	        throw new IllegalArgumentException("Error! emission symbols and emission mat disagree");
	    }     
	    nstate = state.length + 1;
	    this.state = new String[nstate];
	    log_transMat = new double[nstate][nstate];
	    
	    this.state[0] = "Init";        // initial state
	    
	    // P(start -> start) = 0
	    log_transMat[0][0] = Double.NEGATIVE_INFINITY; // = log(0)
	    
	    // P(start -> other) = 1.0/state.length 
	    double fromstart = Math.log(1.0/state.length);
	    
	    for (int j=1; j<nstate; j++)
	      log_transMat[0][j] = fromstart;
	    
	    for (int i=1; i<nstate; i++) {
	      // Reverse state names for efficient backwards concatenation
	      this.state[i] = new StringBuffer(state[i-1]).reverse().toString();
	      // P(other -> start) = 0
	      log_transMat[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
	      for (int j=1; j<nstate; j++)
	        log_transMat[i][j] = Math.log(trans_mat[i-1][j-1]);
	    }
	    
	    // Set up the emission matrix
	    this.Emi_symb = Emi_symb;
	    nEmi_symb = Emi_symb.length();
	    // Assume all Emi_symbs are uppercase letters (ASCII <= 91)
	    log_emiMat = new double[emi_mat.length+1][91];
	    for (int b=0; b<nEmi_symb; b++) {
	      // Use the emitted character, not its number, as index into log_emiMat:
	      char eb = Emi_symb.charAt(b);
	      // P(emit xi in state 0) = 0
	      log_emiMat[0][eb] = Double.NEGATIVE_INFINITY; // = log(0)
	      for (int k=0; k<emi_mat.length; k++) 
	        log_emiMat[k+1][eb] = Math.log(emi_mat[k][b]);
	    }
	  }
	  
	  
	  private static double[] uniformd(int n) {
		    double[] p = new double[n];
		    for (int i=0; i<n; i++) 
		      p[i] = 1.0/n;
		    return p;
		  }    

		  private static double[] randomd(int n) {
		    double[] p = new double[n];
		    double sum = 0;
		    // Generate random numbers
		    for (int i=0; i<n; i++) {
		      p[i] = Math.random();
		      sum += p[i];
		    }
		    // Scale to obtain a discrete probability distribution
		    for (int i=0; i<n; i++) 
		      p[i] /= sum;
		    return p;
		  }
		  
		  private static double fwdbwd(HMM hmm, String[] xs, Forward[] fwds, Backward[] bwds, double[] logP) {
			  double loglikelihood = 0;
			  for (int s=0; s<xs.length; s++) {
				  fwds[s] = new Forward(hmm, xs[s]);
				  bwds[s] = new Backward(hmm, xs[s]);
				  logP[s] = fwds[s].logprob();
				  loglikelihood += logP[s];
			  }
			  return loglikelihood;
		  }
		  
		  	// xs    is the set of training sequences
			// state is the set of HMM state names
			// Emi_symb  is the set of emissible symbols
		  
		public static HMM baumwelch(String[] xs, String[] state, String Emi_symb, final double threshold) {
		  int nstate = state.length;
		  int nseqs  = xs.length;
		  int nEmi_symb  = Emi_symb.length();

		  Forward[] fwds = new Forward[nseqs];
		  Backward[] bwds = new Backward[nseqs];
		  double[] logP = new double[nseqs];
		  
		  double[][] trans_mat = new double[nstate][];
		  double[][] emi_mat = new double[nstate][];

		  // Set up the inverse of b -> Emi_symb.charAt(b); assume all Emi_symbs <= 'Z'
		  int[] Emi_symbinv = new int[91];
		  for (int i=0; i<Emi_symbinv.length; i++)
		    Emi_symbinv[i] = -1;
		  for (int b=0; b<nEmi_symb; b++)
		    Emi_symbinv[Emi_symb.charAt(b)] = b;

		  // Initialize mats with random variables
		  for (int k=0; k<nstate; k++) {
		    trans_mat[k] = randomd(nstate);
		    emi_mat[k] = randomd(nEmi_symb);
		  }

		  HMM hmm = new HMM(state, trans_mat, Emi_symb, emi_mat);
		  
		  double prev_likelihood;

		  // Compute Forward and Backward tables for the sequences
		  double cur_likelihood = fwdbwd(hmm, xs, fwds, bwds, logP);
		  System.out.println("log likelihood = " + cur_likelihood);
		  
		  do {
		    prev_likelihood = cur_likelihood;
		    // Compute estimates for A and E
		    double[][] A = new double[nstate][nstate];
		    double[][] E = new double[nstate][nEmi_symb];
		    for (int s=0; s<nseqs; s++) {
		      String x = xs[s];
		      Forward fwd  = fwds[s];
		      Backward bwd = bwds[s];
		      int L = x.length();
		      double P = logP[s];

		      for (int i=0; i<L; i++) 
		      {
		        for (int k=0; k<nstate; k++) 
		          E[k][Emi_symbinv[x.charAt(i)]] += expo(fwd.f[i+1][k+1] + bwd.b[i+1][k+1] - P);
		      }
		      for (int i=0; i<L-1; i++) 
		        for (int k=0; k<nstate; k++) 
		          for (int ell=0; ell<nstate; ell++) 
		            A[k][ell] += expo(fwd.f[i+1][k+1] + hmm.log_transMat[k+1][ell+1] + hmm.log_emiMat[ell+1][x.charAt(i+1)] + bwd.b[i+2][ell+1] - P);
		    }
		    // Estimate new model parameters
		    for (int k=0; k<nstate; k++) {
		      double Aksum = 0;
		      for (int ell=0; ell<nstate; ell++) 
		        Aksum += A[k][ell];
		      for (int ell=0; ell<nstate; ell++) 
		        trans_mat[k][ell] = A[k][ell] / Aksum;
		      double Eksum = 0;
		      for (int b=0; b<nEmi_symb; b++) 
		        Eksum += E[k][b];
		      for (int b=0; b<nEmi_symb; b++) 
		        emi_mat[k][b] = E[k][b] / Eksum;
		    }
		    // Create new model 
		    hmm = new HMM(state, trans_mat, Emi_symb, emi_mat);
		    
		    cur_likelihood = fwdbwd(hmm, xs, fwds, bwds, logP);
		    System.out.println("log likelihood = " + cur_likelihood);
		    
		    // hmm.print(new SystemOut());
		  } while (Math.abs(prev_likelihood - cur_likelihood) > threshold);
		  
		  return hmm;
		}
		public static double expo(double x) {
		    if (x == Double.NEGATIVE_INFINITY)
		      return 0;
		    else
		      return Math.exp(x);
		  }
	}
class Backward extends HMM_ABS {
	  double[][] b;// the matrix used to find the decoding // b[i][k] = b_k(i) = log(P(x(i+1)..xL, pi_i=k))

	  public Backward(HMM hmm, String x) {
	    super(hmm, x);
	    int L = x.length();
	    b = new double[L+1][hmm.nstate];
	    for (int k=1; k<hmm.nstate; k++)
	      b[L][k] = 0;              // = log(1)  // should be hmm.log_transMat[k][0]
	    for (int i=L-1; i>=1; i--)
	      for (int k=0; k<hmm.nstate; k++) {
	        double sum = Double.NEGATIVE_INFINITY; // = log(0)
	        for (int ell=1; ell<hmm.nstate; ell++) 
	          sum = logLIMIT(sum, hmm.log_transMat[k][ell] 
	                             + hmm.log_emiMat[ell][x.charAt(i)] 
	                             + b[i+1][ell]);
	        b[i][k] = sum;
	      }
	  }
	  double logprob() {
	    double sum = Double.NEGATIVE_INFINITY; // = log(0)
	    for (int ell=0; ell<hmm.nstate; ell++) 
	      sum = logLIMIT(sum, hmm.log_transMat[0][ell] + hmm.log_emiMat[ell][x.charAt(0)]+ b[1][ell]);
	    return sum;
	  }
}
public class LearningProb {
    public static void main(String[] args) {
		    String[] state = { "F", "L" };
		    double[][] aprob = { { 0.95, 0.05 },
		                         { 0.10, 0.90 } };
		    String esym = "123456ABCDEFGH";
		    double[][] eprob = { { 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14,  1.0/14, 1.0/14},
		                         { 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14, 1.0/14,  1.0/14, 1.0/14} };
		    
		    HMM hmm = new HMM(state, aprob, esym, eprob);
		   
		    String x = 
		      "315116246446644245311321631164152133625144543631656626566666"
		    + "ABECDE122345F6ABDFGH14E56EGFH234E56ABEDGH14E23F456ABDFGH1F43"
		    + "222555441666566563564324364131513465146353411126414626253356"
		    + "366163666466232534413661661163252562462255265252266435353336"
		    + "CD1223456ABGH1456GH56AB3456ABDGHCD1223456AB3456ABDGH14423456";
		    
		    String x2 = 
		      "ABECDE122345F6ABDFGH14E56EGFH234E56ABEDGH14E23F456ABDFGH1F43"
		    + "CD1223456ABGH1456GH56AB3456ABDGHCD1223456AB3456ABDGH14423456";

		    String[] xs = { x , x2};
		    HMM estimate = HMM.baumwelch(xs, state, esym, 0.00001);
		    System.out.println("\nTransition probabilities:");
		    for (int i=1; i<estimate.nstate; i++) {
		        for (int j=1; j<estimate.nstate; j++) {
		        	Double xtemp = estimate.log_transMat[i][j];
		        if (xtemp == Double.NEGATIVE_INFINITY)
		        	System.out.print("0.000000 ");
		        else System.out.printf("%.6f ", Math.exp(xtemp));
		        
		      }
		        System.out.println();
		    }
		    
		    System.out.println("\nEmission probabilities:");
		    for (int j=0; j<estimate.nEmi_symb; j++)
		    {
		    	System.out.print(estimate.Emi_symb.charAt(j)+"        ");
		    }System.out.println();
		    for (int i=1; i<estimate.nstate; i++) {
		        for (int j=0; j<estimate.nEmi_symb; j++) {
		        	Double xtemp = estimate.log_emiMat[i][estimate.Emi_symb.charAt(j)];
		        if (xtemp == Double.NEGATIVE_INFINITY)
		        	System.out.print("0.000000 ");
		        else System.out.printf("%.6f ", Math.exp(xtemp));
		        
		      }
		        System.out.println();
		    }
}
}
