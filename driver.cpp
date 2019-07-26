#include "Solution.h"
#include "Subproblem.h"
#include "Extended.h"
#include <stdlib.h>

using namespace std;

double t_quant = 1.282; // For simplicity, just use z quantile (independent of degree of freedom), since sample sizes are usually large in the end

double TIMELIMIT = 7200;

int main(int argc, char** argv)
{
	Solution solution_call;

	// Run program like this:
	// ./adaptiveSamples new_sample_instances/***.dat results/temp 1 0 0.5 1e-3 

	// option: -1 - extensive, 0 - Benders single, 1 - level, 2 - partly inexact bundle defined by partitions, 3 - sequential, 4 - adaptive, 5 - adaptive + partition, 6 - solve instances with a given # of samples in a retrospective way
	// suboption: only apply for option = 3, 4, 5
		// Option = 3: sequential - 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP
		// Option = 4: adaptive (solve SAA up to sample errors) - 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP; 3: fixed rate after B&M fails
		// Option = 5: adaptive + warmup by partition - 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP; 3: fixed rate after B&M fails; 4: fixed rate all over;  5: a heuristic rule to adjust sample size increase rate by some "trust region" rule	
	// Alternative use of suboption for option = 6: # of samples used

	cout << "Importing data..." << endl;
	IloEnv env;
	TSLP prob;

	/*
	prob.firstconstrind = IntMatrix(env);
	prob.firstconstrcoef = NumMatrix(env);
	prob.secondconstrind = IntMatrix(env);
	IloEnv env2;
	prob.secondconstrcoef = NumMatrix(env2);// will create another instance for this
	IloNumArray objcoef(env2);
	objcoef = IloNumArray(env);
	prob.varlb = IloNumArray(env);
	prob.varub = IloNumArray(env);
	prob.firstconstrlb = IloNumArray(env);
	prob.firstconstrub = IloNumArray(env);
	prob.secondconstrlb = IloNumArray(env2);
	prob.secondconstrub = IloNumArray(env2);
	*/
	IloIntArray profile(env);
	prob.objcoef = IloNumArray(env);
	prob.firstvarlb = IloNumArray(env);
	prob.firstvarub = IloNumArray(env);
	prob.secondvarlb = IloNumArray(env);
	prob.secondvarub = IloNumArray(env);
	prob.firstconstrind = IntMatrix(env);
	prob.firstconstrcoef = NumMatrix(env);
	prob.firstconstrlb = IloNumArray(env);
	prob.firstconstrub = IloNumArray(env);
	prob.nbPerRow = IloIntArray(env);
	prob.secondconstrbd = IloNumArray(env);
	prob.secondconstrsense = IloIntArray(env);
	prob.CoefMat = NumMatrix(env);
	prob.CoefInd = IntMatrix(env);
	prob.increaseRate = atof(argv[5]);
	const char* filename = argv[1];
	int option = atoi(argv[3]);
	int suboption = atoi(argv[4]);
	prob.eps = atof(argv[6]);
	prob.distinct_par = 1e-6;
	prob.randomseed = atoi(argv[7]);
	srand(prob.randomseed);


	ifstream file(filename);
	if (!file)
	{
		cerr << "ERROR: could not open file '" << filename
			<< "' for reading" << endl;
		cerr << "usage:  " << argv[0] << " <file>" << endl; //perviously was argv[0]
		throw(-1);
	}
	cout << "before reading..." << endl;
	file >> profile >> prob.objcoef >> prob.firstvarlb >> prob.firstvarub >> prob.secondvarlb >> prob.secondvarub >> prob.firstconstrind >> prob.firstconstrcoef >> prob.firstconstrlb >> prob.firstconstrub >> prob.nbPerRow >> prob.secondconstrbd >> prob.secondconstrsense >> prob.CoefMat >> prob.CoefInd;

	/*
	file >> profile >> objcoef >> prob.firstconstrind >> prob.firstconstrcoef >> prob.secondconstrind >> prob.secondconstrcoef >> prob.varlb >> prob.varub >> prob.firstconstrlb >> prob.firstconstrub >> prob.secondconstrlb >> prob.secondconstrub;
	cout << "after reading..." << endl;
	prob.nbFirstVars = profile[0];
	prob.nbSecVars = profile[1];
	prob.nbScens = profile[2];
	prob.nbRows = profile[3];
	prob.nbFirstRows = prob.firstconstrind.getSize();
	prob.nbSecRows = prob.secondconstrind.getSize();
	prob.CoefMat = NumMatrix(env);
	prob.CoefInd = IntMatrix(env);
	prob.secondconstrbd = IloNumArray(env, prob.nbSecRows);
	prob.secondconstrsense = IloIntArray(env, prob.nbSecRows);
	*/

	cout << "after reading..." << endl;
	prob.nbFirstVars = profile[0];
	prob.nbSecVars = profile[1];
	prob.nbScens = profile[2];
	prob.nbFirstRows = profile[3];
	prob.nbSecRows = profile[4];

	cout << "Number of first-stage vars:" << prob.nbFirstVars << endl;
	cout << "Number of second-stage vars:" << prob.nbSecVars << endl;
	cout << "Number of scenarios:" << prob.nbScens << endl;
	cout << "Number of first rows:" << prob.nbFirstRows << endl;
	cout << "Number of second rows:" << prob.nbSecRows << endl;
	cout << "##################################" << endl;
	solution_call.preprocessing(env, prob);
	/*
	prob.secondconstrcoef.end();
	prob.secondconstrind.end();
	objcoef.end();
	prob.secondconstrub.end();
	prob.secondconstrlb.end();
	env2.end();
	*/
	IloTimer clock(env);
	clock.start();
	STAT stat;
	stat.solvetime = 0;
	stat.mastertime = 0;
	stat.warmstarttime = 0;
	stat.warmstartcuttime = 0;
	stat.qptime = 0;
	stat.subtime = 0;
	stat.relaxobjval = -1e10;
	stat.feasobjval = 1e10;
	stat.evaltime = 0;
	stat.refinetime = 0;
	stat.num_feas_cuts = 0;
	stat.num_opt_cuts = 0;
	stat.iter = 0;
	stat.partitionsize = 0;
	stat.finalpartitionsize = 0;
	if (option == -1)
	{
		// extended formulation

		Extended extend_form;
		extend_form.solve_extended(env, prob, stat, clock);

	}
	if (option == 0)
	{
		// Benders single cut
		vector<int> samples;
		for (int k = 0; k < prob.nbScens; ++k)
			samples.push_back(k);
		VectorXf xiterateXf(prob.nbFirstVars);
		solution_call.solve_singlecut(env, prob, stat, clock, samples, xiterateXf);
		cout << "relaxobjval = " << stat.relaxobjval << endl;
		cout << "feasobjval = " << stat.feasobjval << endl;
		cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
	}
	if (option == 1)
	{
		// Level method
		vector<int> samples;
		for (int k = 0; k < prob.nbScens; ++k)
			samples.push_back(k);
		VectorXf xiterateXf(prob.nbFirstVars);
		solution_call.solve_level(env, prob, stat, clock, samples, xiterateXf, 0);
		cout << "relaxobjval = " << stat.relaxobjval << endl;
		cout << "feasobjval = " << stat.feasobjval << endl;
		cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
	}
	

	if (option == 2 || option == 6)
	{
		// partily inexact bundle method with partitions
		prob.kappa = 0.8;
		prob.kappaf = 0.3;
		vector<int> samples;
		if (option == 2)
		{
			for (int k = 0; k < prob.nbScens; ++k)
				samples.push_back(k);
		}
		if (option == 6)
		{
			int nSampleUsed = atoi(argv[4]);
			for (int j = 0; j < nSampleUsed; ++j)
				samples.push_back(rand() % prob.nbScens);
		}
		VectorXf xiterateXf(prob.nbFirstVars);

		Subproblem sub_call(env, prob);
		sub_call.construct_second_opt(env, prob);
		sub_call.construct_second_feas(env, prob);

		vector<DualInfo> dualInfoCollection;
		vector<VectorXf> rhsvecs;
		// Store all the rhs vectors
		for (int k = 0; k < prob.nbScens; ++k)
		{
			VectorXf rhsXf(prob.nbSecRows);
			for (int j = 0; j < prob.nbSecRows; ++j)
				rhsXf[j] = prob.secondconstrbd[j + k * prob.nbSecRows];
			rhsvecs.push_back(rhsXf);
		}
		int nearOptimal = 0;
		bool tempflag = solution_call.solve_partly_inexact_bundle(env, prob, stat, clock, samples, xiterateXf, 0, 1, dualInfoCollection, rhsvecs, nearOptimal, TIMELIMIT);
		cout << "xiterateXf = ";
		for (int j = 0; j < prob.nbFirstVars; ++j)
			cout << xiterateXf[j] << " ";
		cout << endl;
	}
	/*if (option == 3)
	{
		// Sequential procedure: solve SAAs in each iteration to optimality
		// 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP
		solve_adaptive(env, prob, stat, clock, suboption, 0);
	}
	if (option == 4)
	{
		// Adaptive procedure: solve SAAs in each iteration to the sampling error
		// 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP; 3: fixed rate schedule after BM's schedule fails
		solve_adaptive(env, prob, stat, clock, suboption, 1);
	}
	if (option == 5)
	{
		// Adaptive procedure, use partition-based approach (with warm starts) to solve the sampled problems
		prob.kappa = 0.8;
		prob.kappaf = 0.3;
		solve_adaptive_partition(env, prob, stat, clock, suboption);
	}*/
	cout << "solvetime = " << stat.solvetime << endl;

	ofstream out(argv[2], ios::app);
	if (out) {
		out << setw(16) << argv[1];
		out << setw(8) << argv[3];
		if (option != 3 && option != 4 && option != 5)
		{
			out << setw(12) << stat.relaxobjval;
			out << setw(12) << stat.feasobjval;
			out << setw(12) << stat.subtime;
		}
		out << setw(8) << stat.solvetime;
		out << setw(8) << stat.iter;
		if (option == 3 || option == 4 || option == 5)
		{
			out << setw(8) << stat.mainIter;
			out << setw(8) << stat.finalSampleSize;
			out << setw(12) << stat.finalSolExactObj;
			out << setw(12) << stat.gapThreshold;
			if (option == 5)
			{
				out << setw(12) << stat.warmstarttime;
				out << setw(12) << stat.warmstartcuttime;
			}
			out << setw(12) << stat.mastertime;
			out << setw(12) << stat.evaltime;
		}
		out << endl;
		out.close();
	}
	env.end();
	return 0;
}
