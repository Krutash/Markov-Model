import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;

class ProbVector{
	ArrayList<Double>probs = new ArrayList<>();
	ArrayList<String> states = new ArrayList<>();
	Hashtable<String, Double> PV = new Hashtable<>();
	public ProbVector(Hashtable<String, Double> mydict) {
		Set<String> keys = mydict.keySet();
		Iterator<String> itr2 = keys.iterator();
		while(itr2.hasNext())
		{
			String key = itr2.next();
			probs.add(mydict.get(key));
			states.add(key);
		}

		if(probs.size() != states.size()) 
		{System.out.println("The probabilites must match the states"); System.exit(1);}
		Double temp =(double) 0; 
		Iterator<Double> itr = probs.iterator();
		while(itr.hasNext())
			temp = temp + itr.next();
		
		if(Math.abs(temp -1) >1e-12) {System.out.println("Error ! Probabilities must add up to 1");System.exit(1);}
		this.PV = mydict;
		//printProbs();
	}
	public void printProbs() {
		System.out.println(PV);
	}
}

class ProbMatrix {
	ArrayList<String> statesRow = new ArrayList<>();
	Hashtable<String, ArrayList<Double>> PM = new Hashtable<String, ArrayList<Double>>();
	ArrayList<String> statesColumn = new ArrayList<>();
	Hashtable<String, ProbVector> P = new Hashtable<>();
	public ProbMatrix(Hashtable<String, ProbVector> PM) {
		Set<String> keys = PM.keySet();
		Iterator<String> itr = keys.iterator();
		statesRow = PM.get(itr.next()).states;
		while(itr.hasNext()) {
			Iterator<String> temp = PM.get(itr.next()).states.iterator();
			Iterator<String> pvKeys = statesRow.iterator();
			while(temp.hasNext() && pvKeys.hasNext()){
				if(temp.next().compareTo(pvKeys.next())!=0)
				{System.out.println("Error! Please check spellings or Insufficent Arguments"); System.exit(1);}
			}
			if(temp.hasNext() || pvKeys.hasNext()) {
				System.out.println("Error! Please check spellings or Insufficent Arguments"); System.exit(1);
			}
		}
		this.P = PM;
		itr = keys.iterator();
		while(itr.hasNext()){
			String name = itr.next();
			this.statesColumn.add(name);
			ProbVector temp = PM.get(name);
			ArrayList<Double> temp2 = new ArrayList<>();
			temp2 = temp.probs;
			this.PM.put(name, temp2);	
		}
		//System.out.println(this.PM);
	}
}

class Viterbi
{
	
	public ArrayList<String> compute(ArrayList<String> obs, ArrayList<String> states, ProbVector start_prob, ProbMatrix trans_prob, ProbMatrix emiss_prob) throws ArrayIndexOutOfBoundsException
	{
		double[][] viterbi = new double[obs.size()][states.size()];
		int [][] path = new int[states.size()][obs.size()];

		// Viterbi matrix initializing.
		int st = 0;

		ArrayList<String> emissionStates = emiss_prob.statesRow;
		int FirstemissionIndex = emissionStates.indexOf(obs.get(0));
		for(String state : states)
		{	
			ArrayList<Double> emissionsforState = emiss_prob.PM.get(state);
			viterbi[0][st] = start_prob.PV.get(state)*emissionsforState.get(FirstemissionIndex);
			path[st][0] = st;
			st++;
		}
		
		for(int i=1; i<obs.size(); i++)
		{
			int [][] newpath = new int[states.size()][obs.size()];
			int CurSt = 0;
			int emissionIndex = emissionStates.indexOf(obs.get(i));
			for(String cur_state : states)
			{
				double prob = -1.0;
				String state;
				int FromSt = 0;
				int tempIndexOfTrans = trans_prob.statesRow.indexOf(cur_state);
				for(String from_state: states)
				{
					ArrayList<Double> transForState = trans_prob.PM.get(from_state);
					ArrayList<Double> emissionsforState = emiss_prob.PM.get(cur_state);
					
					double nprob = viterbi[i - 1][FromSt] * transForState.get(tempIndexOfTrans) * emissionsforState.get(emissionIndex);
					if(nprob > prob)
					{
						// Re-assign, if only greater.
						prob = nprob;
						state = from_state;
						viterbi[i][CurSt] = prob;
						System.arraycopy(path[states.indexOf(state)], 0, newpath[CurSt], 0, i);
						newpath[CurSt][i] = CurSt;
						
					}
					FromSt++;
				}
				CurSt++;
			}
			path = newpath;
				
		}
		
		double prob = -1;
        int state = 0;

		// The final path computataion.
        int stemp = 0;
		for(String state1: states)
		{
			if(viterbi[obs.size()-1][stemp] > prob)
			{
				prob = viterbi[obs.size()-1][stemp];
				state = stemp;
			}
			stemp++;
		}
		
		ArrayList<String> pathres = new ArrayList<>();
		
		for(int i : path[state])
		{
			pathres.add(states.get(i));
		}
		return pathres;
	}
}

class Forward
{
	double forward[][];
	
	public double compute(ArrayList<String> obs,  ArrayList<String>states, ProbVector start_prob, ProbMatrix trans_prob, ProbMatrix emiss_prob) throws ArrayIndexOutOfBoundsException
	{
		forward = new double[obs.size()][states.size()];
		
		// Initializing the Forward Matrix
		int st = 0;
		ArrayList<String> emissionStates = emiss_prob.statesRow;
		int FirstemissionIndex = emissionStates.indexOf(obs.get(0));
		for(String state : states)
		{	
			ArrayList<Double> emissionsforState = emiss_prob.PM.get(state);
			forward[0][st] = start_prob.PV.get(state)*emissionsforState.get(FirstemissionIndex);
			st++;
		}
		
		st = 0;
		
		for(int i=1; i<obs.size(); i++)
		{
			st = 0;
			for(String state1 : states)
			{
				forward[i][st] = 0;
				int stTemp = 0;
				int tempIndexOfTrans = trans_prob.statesRow.indexOf(state1);
				for(String state2 : states)
				{
					ArrayList<Double> transForState = trans_prob.PM.get(state2);
					forward[i][st] += forward[i - 1][stTemp] * transForState.get(tempIndexOfTrans);
					
					// Forward Algorithm adds up every probability calculated, takes to the maximum.
					stTemp++;
				}
				ArrayList<Double> emissionsState = emiss_prob.PM.get(state1);
				int emissionIndex = emissionStates.indexOf(obs.get(i));
				forward[i][st] *= emissionsState.get(emissionIndex);
				st++;
			}
		}
		
		// To check the status of Forward Matrix.
		
		
		// Calculation of final likelihood probability.
		double prob = 0;
		for(int i = 0; i< states.size(); i++)
		{
			prob += forward[obs.size() - 1][i];
		}
		
		return prob;
	}

}

class HMM {
	ProbMatrix T;
	ProbMatrix E;
	ProbVector pi;
	ArrayList<String> Obs;
	ArrayList<String> Hidden;
	public HMM(ProbMatrix T, ProbMatrix E, ProbVector pi) {
		this.T = T;
		this.E = E;
		this.pi = pi;
		this.Obs = E.statesRow;
		this.Hidden = pi.states;
	}

}

public class HiddenMarkov {

	
	public static void main(String[] args)
	{
		
		int N = 0; //hidden states
		int M = 0; //number of observables
		Scanner in = new Scanner(System.in);
		System.out.print("Number of latent (hidden) states :");
		N = in.nextInt();
		System.out.print("give values for Initialization Matrix:\n(For example : sun 0.1 rain 0.5 cloud 0.4): ");
		Hashtable<String, Double>piHash = new Hashtable<>();
		for(int i = 0; i< N; i++)
		{
			piHash.put(in.next(), in.nextDouble());
		}
		ProbVector pi = new ProbVector(piHash);
		System.out.print("\ngive values for Transition Matrix:\n(For example matrix row-wise as:\n"
				+ "\nsun sun 0.1 rain 0.5 cloud 0.4"
				+ "\nrain sun 0.3 rain 0.3 cloud 0.4\ncloud sun 0.4 rain 0.4 cloud 0.2)\n:>");
		Hashtable<String,  ProbVector> tmTab = new Hashtable<>();
		for(int i = 0 ; i<N; i++)
		{
			String state = in.next();
			Hashtable<String, Double> tmHash = new Hashtable<>();
			for(int j = 0; j< N; j++)
			{
				tmHash.put(in.next(), in.nextDouble());
			}
			ProbVector tmLine = new ProbVector(tmHash);
			tmTab.put(state, tmLine);
			System.out.print(":>");
		}
		ProbMatrix TM = new ProbMatrix(tmTab);
		
		System.out.print("\nNumber of observables :");
		M = in.nextInt();
		
		System.out.print("\ngive values for Emission Matrix:\n(For example matrix row-wise as:\n"
				+ "\nsun playOut 0.1 clean 0.5 study 0.4"
				+ "\nrain playOut 0.3 clean 0.3 study 0.4\ncloud playOut 0.4 clean 0.4 study 0.2)\n:>");
		
		Hashtable<String,  ProbVector> emTab = new Hashtable<>();
		for(int i = 0 ; i<N; i++)
		{
			String state = in.next();
			Hashtable<String, Double> tmHash = new Hashtable<>();
			for(int j = 0; j< M; j++)
			{
				tmHash.put(in.next(), in.nextDouble());
			}
			ProbVector tmLine = new ProbVector(tmHash);
			emTab.put(state, tmLine);
			System.out.print(":>");
		}
		ProbMatrix EM = new ProbMatrix(emTab);
		
		HMM model = new HMM(TM, EM, pi);
		
		//T = in.nextInt();
		System.out.print("Select one of the following:\n1. Likelihood Problem\n2. Decoding Problem\n0. exit\nYour Choice :>");
		int choice = in.nextInt();
		
		while(choice != 0 && (choice == 1 || choice == 2))
		{
		if(choice == 1)
		{
			System.out.print("Length of sequence :");
			int ln = in.nextInt();
			ArrayList<String> seq = new ArrayList<String>();
			System.out.print("\nProvide the seq (for example : playOut study study clean):");
			for(int i =0; i<ln; i++)
			{
				seq.add(in.next());
			}
			
			Forward likli = new Forward();
			double prob = likli.compute(seq, model.Hidden, model.pi, model.T, model.E);
			System.out.print("Likelihood : ");
			System.out.println(prob);
		}
		if(choice == 2)
		{
			System.out.print("Length of sequence :");
			int ln = in.nextInt();
			ArrayList<String> seq = new ArrayList<String>();
			System.out.print("\nProvide the seq (for example : playOut study study clean):");
			for(int i =0; i<ln; i++)
			{
				seq.add(in.next());
			}
			
			Viterbi decode = new Viterbi();
			ArrayList<String> path = decode.compute(seq, model.Hidden, model.pi, model.T, model.E);
			
			System.out.print("Decoded sequnce of hidden states: ");
			System.out.println(path);
		}
		System.out.print("Select one of the following:\n1. Likelihood Problem\n2. Decoding Problem\n0. exit\nYour Choice :>");
		choice = in.nextInt();
		}
		
		in.close();
	}
	
}