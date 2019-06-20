/*- mode: C++;
 * Date: Jan. 8, 2019
  // Note: recently changed sequentialSetup function; got rid of evalFlag; added \delta = 1e-3; changed the adaptive rate selection
 */

#include <iostream>
#include <ilcplex/ilocplex.h>
#include "adaptiveSamples.h"
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
extern "C"{
#include <stdio.h>
#include <stdlib.h>
}
#include <time.h>

#include "Extended.h"

using namespace std;
//using Eigen::MatrixXf;
//using Eigen::VectorXf;

double t_quant = 1.282; // For simplicity, just use z quantile (independent of degree of freedom), since sample sizes are usually large in the end

double TIMELIMIT = 7200;

int main(int argc, char **argv)
{
	// Run program like this:

	// ./adaptiveSamples new_sample_instances/***.dat results/temp 1 0 0.5 1e-3

	// ./adaptiveSamples new_sample_instances/***.dat results/temp 1 0 0.5 1e-3 

	/*
	"C:\Users\3deva\source\repos\adaptiveSamples\x64\Release\adaptiveSamples.exe" 
	"C:\Users\3deva\source\repos\adaptiveSamples\instances\20x20-1-20000-1-clean.dat" 
	"C:\Users\3deva\source\repos\adaptiveSamples\results\temp" -1 0 0.5 1e-3 5
	*/

	// option: -1 - extensive, 0 - Benders single, 1 - level, 2 - partly inexact bundle defined by partitions, 3 - sequential, 4 - adaptive, 5 - adaptive + partition, 6 - solve instances with a given # of samples in a retrospective way
	// suboption: only apply for option = 3, 4, 5
		// Option = 3: sequential - 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP
		// Option = 4: adaptive (solve SAA up to sample errors) - 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP; 3: fixed rate after B&M fails
   		// Option = 5: adaptive + warmup by partition - 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP; 3: fixed rate after B&M fails; 4: fixed rate all over;  5: a heuristic rule to adjust sample size increase rate by some "trust region" rule
	// Alternative use of suboption for option = 6: # of samples used

    cout << "Importing data..." << endl;
	//cout << "Hello\n";
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

	/*
	cout << "argv[0] : " << argv[0] << endl;
	cout << "argv[1] : " << argv[1] << endl;
	cout << "argv[2] : " << argv[2] << endl;
	cout << "argv[3] : " << argv[3] << endl;
	cout << "argv[4] : " << argv[4] << endl;
	cout << "argv[5] : " << argv[5] << endl;
	cout << "argv[6] : " << argv[6] << endl;
	cout << "argv[7] : " << argv[7] << endl;
	*/

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
	preprocessing(env, prob);
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
		// solve_extended(env, prob, stat, clock);
	
		Extended extend_form(env, prob, stat, clock);
		extend_form.solve_extended;

	}
	if (option == 0)
	{
		// Benders single cut
		vector<int> samples;
		for (int k = 0; k < prob.nbScens; ++k)
			samples.push_back(k);
		VectorXf xiterateXf(prob.nbFirstVars);
		solve_singlecut(env, prob, stat, clock, samples, xiterateXf);
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
		solve_level(env, prob, stat, clock, samples, xiterateXf, 0);
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
		Subprob subp;
		construct_second_opt(env, prob, subp);
		construct_second_feas(env, prob, subp);
		vector<DualInfo> dualInfoCollection;
		vector<VectorXf> rhsvecs;
		// Store all the rhs vectors
		for (int k = 0; k < prob.nbScens; ++k)
		{
			VectorXf rhsXf(prob.nbSecRows);
			for (int j = 0; j < prob.nbSecRows; ++j)
				rhsXf[j] = prob.secondconstrbd[j+k*prob.nbSecRows];
			rhsvecs.push_back(rhsXf);
		}
		int nearOptimal = 0;
		bool tempflag = solve_partly_inexact_bundle(env, prob, subp, stat, clock, samples, xiterateXf, 0, 1, dualInfoCollection, rhsvecs, nearOptimal, TIMELIMIT);
		cout << "xiterateXf = ";
		for (int j = 0; j < prob.nbFirstVars; ++j)
			cout << xiterateXf[j] << " ";
		cout << endl;
		subp.suboptcplex.end();
		subp.suboptmodel.end();
		subp.suboptcon.end();
		subp.subopty.end();
		subp.subfeascplex.end();
		subp.subfeasmodel.end();
		subp.subfeascon.end();
		subp.subfeasy.end();
	}
	if (option == 3)
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
	}
	cout << "solvetime = " << stat.solvetime << endl;

	ofstream out(argv[2], ios::app);
	if (out)  {
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

double subprob(Subprob& subp, const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, int k, bool& feasflag)
{
	// First try solving the optimization model
	// Set constraint bounds
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		double bd = prob.secondconstrbd[k*prob.nbSecRows+i];
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind < prob.nbFirstVars)
				bd -= prob.CoefMat[i][j]*xvals[ind];
		}
		if (prob.secondconstrsense[i] == -1)
			subp.suboptcon[i].setLB(bd);
		if (prob.secondconstrsense[i] == 0)
		{
			if (subp.suboptcon[i].getUB() < bd)
			{
				subp.suboptcon[i].setUB(bd);
				subp.suboptcon[i].setLB(bd);
			}
			else
			{
				subp.suboptcon[i].setLB(bd);
				subp.suboptcon[i].setUB(bd);
			}
		}
		if (prob.secondconstrsense[i] == 1)
			subp.suboptcon[i].setUB(bd);
	}
	// Set variable bounds
	for (int j = 0; j < prob.nbSecVars; ++j)
	{
		if (prob.secondvarlb[k*prob.nbSecVars+j] != -IloInfinity)
			subp.suboptcon[prob.nbSecRows+j].setLB(prob.secondvarlb[k*prob.nbSecVars+j]);
		else
			subp.suboptcon[prob.nbSecRows+j].setLB(-IloInfinity);
		if (prob.secondvarub[k*prob.nbSecVars+j] != IloInfinity)
			subp.suboptcon[prob.nbSecRows+prob.nbSecVars+j].setLB(-prob.secondvarub[k*prob.nbSecVars+j]);
		else
			subp.suboptcon[prob.nbSecRows+prob.nbSecVars+j].setLB(-IloInfinity);
	}
	subp.suboptcplex.solve();
	double returnval;
	if (subp.suboptcplex.getStatus() == IloAlgorithm::Optimal)
	{
		returnval = subp.suboptcplex.getObjValue();
		subp.suboptcplex.getDuals(duals, subp.suboptcon);
		feasflag = 1;
	}
	else
	{
		feasflag = 0;
		// infeasible! Get extreme rays
		for (int i = 0; i < prob.nbSecRows; ++i)
		{
			double bd = prob.secondconstrbd[k*prob.nbSecRows+i];
			for (int j = 0; j < prob.nbPerRow[i]; ++j)
			{
				int ind = prob.CoefInd[i][j];
				if (ind < prob.nbFirstVars)
					bd -= prob.CoefMat[i][j]*xvals[ind];
			}
			if (prob.secondconstrsense[i] == -1)
				subp.subfeascon[i].setLB(bd);
			if (prob.secondconstrsense[i] == 0)
			{
				if (subp.subfeascon[i].getUB() < bd)
				{
					subp.subfeascon[i].setUB(bd);
					subp.subfeascon[i].setLB(bd);
				}
				else
				{
					subp.subfeascon[i].setLB(bd);
					subp.subfeascon[i].setUB(bd);
				}
			}
			if (prob.secondconstrsense[i] == 1)
				subp.subfeascon[i].setUB(bd);
		}
		for (int j = 0; j < prob.nbSecVars; ++j)
		{
			if (prob.secondvarlb[k*prob.nbSecVars+j] != -IloInfinity)
				subp.subfeascon[prob.nbSecRows+j].setLB(prob.secondvarlb[k*prob.nbSecVars+j]);
			else
				subp.subfeascon[prob.nbSecRows+j].setLB(-IloInfinity);
			if (prob.secondvarub[k*prob.nbSecVars+j] != IloInfinity)
				subp.subfeascon[prob.nbSecRows+prob.nbSecVars+j].setLB(-prob.secondvarub[k*prob.nbSecVars+j]);
			else
				subp.subfeascon[prob.nbSecRows+prob.nbSecVars+j].setLB(-IloInfinity);
		}
		subp.subfeascplex.solve();
		subp.subfeascplex.getDuals(duals, subp.subfeascon);
		returnval = subp.subfeascplex.getObjValue();
	}
	return returnval;
}

void gen_feasibility_cuts(IloEnv& env, const TSLP& prob, const IloNumArray& xvals, const vector<int>& extreme_ray_map, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, const double sum_of_infeas, IloModel& model, const IloNumVarArray& x)
{
	vector<double> feas_cut_coef(prob.nbFirstVars, 0);
	double sum_xvals = 0.0;
	for (int j = 0; j < extreme_ray_map.size(); ++j)
	{
		int ind = extreme_ray_map[j];
		for (int i = 0; i < prob.nbSecRows; ++i)
		{
			for (int j = 0; j < prob.nbPerRow[i]; ++j)
			{
				if (prob.CoefInd[i][j] < prob.nbFirstVars)
				{
					feas_cut_coef[prob.CoefInd[i][j]] += prob.CoefMat[i][j]*extreme_rays[ind][i];
					sum_xvals += prob.CoefMat[i][j]*extreme_rays[ind][i]*xvals[prob.CoefInd[i][j]];
				}
			}
		}
	}
	IloExpr lhs(env);
	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		if (fabs(feas_cut_coef[j]) > 1e-7)
			lhs += x[j]*feas_cut_coef[j];
	}
	model.add(lhs >= sum_of_infeas+sum_xvals);
	lhs.end();
}

void preprocessing(IloEnv& env, TSLP& prob)
{
	// Rearrange the second-stage constraints data and variable bound data s.t.
	// 1 - eliminate unnecessary constraints like <= IloInfinity
	// 2 - all constraints are <=, >= or = constraints, if both bounds exist, append to the end of the constraint list
	// 3 - all variables are >= are = constraints
	for (int j = 0; j < prob.firstvarlb.getSize(); ++j)
	{
		if (prob.firstvarlb[j] == -1e10)
			prob.firstvarlb[j] = -IloInfinity;
		if (prob.firstvarub[j] == 1e10)
			prob.firstvarub[j] = IloInfinity;
	}

	for (int j = 0; j < prob.secondvarlb.getSize(); ++j)
	{
		if (prob.secondvarlb[j] == -1e10)
			prob.secondvarlb[j] = -IloInfinity;
		if (prob.secondvarub[j] == 1e10)
			prob.secondvarub[j] = IloInfinity;
	}

	// Construct CoefMatXf:
	prob.CoefMatXf = MatrixXf(prob.nbSecRows, prob.nbFirstVars);
	prob.CoefMatXf.setConstant(0);
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind < prob.nbFirstVars)
				prob.CoefMatXf(i,ind) = prob.CoefMat[i][j];
		}
	}
}

double solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples)
{
	IloModel meanmodel(meanenv);
	IloNumVarArray meanx(meanenv, prob.firstvarlb, prob.firstvarub);
	IloNumVarArray meany(meanenv, prob.nbSecVars, -IloInfinity, IloInfinity);
	// first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(meanenv);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += meanx[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(meanenv, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		meanmodel.add(range);
		lhs.end();
	}
	int nbScens = samples.size();
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(meanenv);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs += meany[ind-prob.nbFirstVars]*prob.CoefMat[i][j];
			else
				lhs += meanx[ind]*prob.CoefMat[i][j];
		}
		// set constraint bounds
		double bd = 0;
		for (int k = 0; k < nbScens; ++k)
			bd += prob.secondconstrbd[samples[k]*prob.nbSecRows+i];
		bd = bd*1.0/nbScens;
		IloRange range;
		if (prob.secondconstrsense[i] == -1)
			range = IloRange(meanenv, bd, lhs, IloInfinity);
		if (prob.secondconstrsense[i] == 0)
			range = IloRange(meanenv, bd, lhs, bd);
		if (prob.secondconstrsense[i] == 1)
			range = IloRange(meanenv, -IloInfinity, lhs, bd);
		meanmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(meanenv);
		lhs += meany[i];
		IloRange range(meanenv, -IloInfinity, lhs, IloInfinity);
		if (prob.secondvarlb[i] != -IloInfinity)
		{
			double bd = 0;
			for (int k = 0; k < nbScens; ++k)
				bd += prob.secondvarlb[samples[k]*prob.nbSecVars+i];
			bd = bd*1.0/nbScens;
			range.setLB(bd);
		}
		if (prob.secondvarub[i] != IloInfinity)
		{
			double bd = 0;
			for (int k = 0; k < nbScens; ++k)
				bd += prob.secondvarub[samples[k]*prob.nbSecVars+i];
			bd = bd*1.0/nbScens;
			range.setUB(bd);
		}
		meanmodel.add(range);
		lhs.end();
	}
	IloExpr meanobj(meanenv);
	for (int i = 0; i < prob.nbFirstVars; ++i)
		meanobj += meanx[i]*prob.objcoef[i];

	for (int i = 0; i < prob.objcoef.getSize(); ++i)
	{
		if (i >= prob.nbFirstVars)
			meanobj += meany[i-prob.nbFirstVars]*prob.objcoef[i];
	}
	meanmodel.add(IloMinimize(meanenv, meanobj));
	meanobj.end();
	IloCplex meancplex(meanmodel);
	meancplex.setParam(IloCplex::TiLim, 3600);
	meancplex.setParam(IloCplex::Threads, 1);
	meancplex.setParam(IloCplex::SimDisplay, 0);
	meancplex.setOut(meanenv.getNullStream());
	meancplex.solve();
	meancplex.getValues(meanxvals, meanx);
	double returnval = meancplex.getObjValue();
	meancplex.end();
	meanmodel.end();
	meanx.end();
	meany.end();
	return returnval;
}
/*
void solve_extended(IloEnv& env, const TSLP& prob, STAT& stat, IloTimer& clock)
{
	IloModel model(env);
	Model mod;
	mod.x = IloNumVarArray(env, prob.firstvarlb, prob.firstvarub);
	mod.y = IloNumVarArray(env, prob.secondvarlb, prob.secondvarub);
	// First-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += mod.x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		model.add(range);
		lhs.end();
	}
	// Second-stage constraints
	for (int k = 0; k < prob.nbScens; ++k)
	{
		for (int i = 0; i < prob.nbSecRows; ++i)
		{
			IloExpr lhs(env);
			for (int j = 0; j < prob.nbPerRow[i]; ++j)
			{
				int ind = prob.CoefInd[i][j];
				if (ind < prob.nbFirstVars)
					lhs += mod.x[ind]*prob.CoefMat[i][j];
				else
					lhs += mod.y[k*prob.nbSecVars+ind-prob.nbFirstVars]*prob.CoefMat[i][j];
			}
			IloRange range;
			if (prob.secondconstrsense[i] == -1)
			{
				// -1: >=, 0: =, 1: <=
				range = IloRange(env, prob.secondconstrbd[k*prob.nbSecRows+i], lhs, IloInfinity);
			}
			if (prob.secondconstrsense[i] == 1)
				range = IloRange(env, -IloInfinity, lhs, prob.secondconstrbd[k*prob.nbSecRows+i]);
			if (prob.secondconstrsense[i] == 0)
				range = IloRange(env, prob.secondconstrbd[k*prob.nbSecRows+i], lhs, prob.secondconstrbd[k*prob.nbSecRows+i]);
			model.add(range);
			lhs.end();
		}
	}
	// Objective
	IloExpr obj(env);
	for (int i = 0; i < prob.objcoef.getSize(); ++i)
	{
		// The reason why we need to have i and ind separately is that, we have some variables that don't have obj coef, i.e. 0 coef
		if (i < prob.nbFirstVars)
		{
			// meaning it is a first-stage variable
			obj += mod.x[i]*prob.objcoef[i];
		}
		else
		{
			// meaning it is a second-stage variable
			// For now let's assume coef is the same for each scenario
			for (int k = 0; k < prob.nbScens; ++k)
			{
				double coef = prob.objcoef[i]*1.0/prob.nbScens;
				obj += mod.y[i-prob.nbFirstVars+k*prob.nbSecVars]*coef;
			}
		}
	}
	model.add(IloMinimize(env, obj));
	obj.end();
	IloCplex cplex(model);
	cplex.setParam(IloCplex::TiLim,10800);
	cplex.setParam(IloCplex::Threads, 1);
	// Barrier
	cplex.setParam(IloCplex::RootAlg, 4);
	cplex.setParam(IloCplex::BarDisplay, 0);
	//cplex.setParam(IloCplex::EpOpt, 1e-6);

	double lasttime = clock.getTime();
	cplex.solve();
	stat.solvetime = clock.getTime() - lasttime;
	// get solution info
	stat.relaxobjval = cplex.getObjValue();
	stat.feasobjval = cplex.getObjValue();
	cplex.end();
	model.end();
	mod.x.end();
	mod.y.end();
}
*/
void solve_singlecut(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf)
{
	// Benders: single cut
	double starttime = clock.getTime();
	bool feas_flag = 1;
	IloModel mastermodel(env);
	IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
	// For now we assume an LB for the second-stage obj: 0
	IloNumVar theta(env, 0, IloInfinity);
	// Adding first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		mastermodel.add(range);
		lhs.end();
	}
	// Adding objective
	IloExpr obj(env);
	for (int i = 0; i < prob.nbFirstVars; ++i)
		obj += x[i]*prob.objcoef[i];
	// single cut
	obj += theta;
	mastermodel.add(IloMinimize(env, obj));
	obj.end();
	IloCplex mastercplex(mastermodel);
	mastercplex.setParam(IloCplex::TiLim,10800);
	mastercplex.setParam(IloCplex::Threads, 1);
	// Barrier
	mastercplex.setParam(IloCplex::RootAlg, 2);
	mastercplex.setParam(IloCplex::BarDisplay, 0);
	mastercplex.setParam(IloCplex::SimDisplay, 0);
	mastercplex.setOut(env.getNullStream());
	//mastercplex.setParam(IloCplex::EpOpt, 1e-4);

	// We initiate an LP model for the second-stage problem, and everytime (iteration/sceanrio) we just update the rhs and solve: constraint coefficients are the same across all scenarios
	// we initiate both optimization model and feasibility model
	Subprob subp;
	construct_second_opt(env, prob, subp);
	construct_second_feas(env, prob, subp);

	int nbScens = samples.size();

	while ((stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) > 1e-6 || feas_flag == 0)
	{
		feas_flag = 1;
		stat.iter++;
		IloNumArray xvals(env);
		double thetaval;
		double lasttime = clock.getTime();
		mastercplex.solve();
		cout << "master status = " << mastercplex.getStatus() << endl;
		if (mastercplex.getStatus() == IloAlgorithm::Unbounded)
		{
			IloEnv meanenv;
			stat.relaxobjval = solve_mean_value_model(prob, meanenv, xvals, samples);
			meanenv.end();
		}
		else
		{
			stat.relaxobjval = mastercplex.getObjValue();
			mastercplex.getValues(xvals, x);
			thetaval = mastercplex.getValue(theta);
		}
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterateXf(j) = xvals[j];
		stat.mastertime += clock.getTime()-lasttime;
		double feasbound = 0.0;
		IloExpr lhsaggr(env);
		double lhsaggrval = 0;
		lhsaggr += theta;
		lhsaggrval += thetaval;
		double rhsaggr = 0;
		lasttime = clock.getTime();
		for (int k = 0; k < nbScens; ++k)
		{
			// solve subproblems for each scenario
			IloNumArray duals(env);
			bool feasflag;
			double subobjval = subprob(subp, prob, xvals, duals, samples[k], feasflag);
			VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
			for (int i = 0; i < prob.nbSecRows+prob.nbSecVars; ++i)
				dualvec(i) = duals[i];
			duals.end();
			if (feasflag == 1)
			{
				// optimal, so return extreme point solution
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
				double sum_xvals = 0;
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
					{
						lhsaggrval += xvals[j]*opt_cut_coef[j]*1.0/nbScens;
						lhsaggr += x[j]*opt_cut_coef[j]*1.0/nbScens;
						sum_xvals += opt_cut_coef[j]*xvals[j];
					}
				}
				double rhssub = subobjval + sum_xvals;
				rhsaggr += rhssub*1.0/nbScens;
				feasbound += subobjval;
			}
			else
			{
				cout << "infeasible!" << endl;
				feas_flag = 0;
				// infeasible, so return extreme rays
				VectorXf feas_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = feas_cut_coef.dot(xiterateXf);
				IloExpr lhssub(env);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(feas_cut_coef[j]) > 1e-7)
						lhssub += x[j]*feas_cut_coef[j];
				}
				double rhssub = sum_xvals+subobjval;
				stat.num_feas_cuts++;
				mastermodel.add(lhssub >= rhssub);
				lhssub.end();
			}
		}
		stat.subtime += clock.getTime()-lasttime;
		if (feas_flag == 1)
		{
			feasbound = feasbound*1.0/nbScens;
			for (int j = 0; j < prob.nbFirstVars; ++j)
				feasbound += xvals[j]*prob.objcoef[j];
			if (feasbound <= stat.feasobjval)
				stat.feasobjval = feasbound;
			mastermodel.add(lhsaggr >= rhsaggr);
			stat.num_opt_cuts++;
		}
		else
			cout << "Infeasible! Generating feasibility cuts!" << endl;
		lhsaggr.end();
		xvals.end();
		cout << "relaxobjval = " << stat.relaxobjval << endl;
		cout << "feasobjval = " << stat.feasobjval << endl;
		cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
		stat.solvetime = clock.getTime()-starttime;
		cout << "stat.solvetime = " << stat.solvetime << endl;
		if (stat.solvetime > 10800)
			break;
	}
	subp.suboptcplex.end();
	subp.suboptmodel.end();
	subp.suboptcon.end();
	subp.subopty.end();
	subp.subfeascplex.end();
	subp.subfeasmodel.end();
	subp.subfeascon.end();
	subp.subfeasy.end();
	mastercplex.end();
	mastermodel.end();
	x.end();
}

void solve_level(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option)
{
	// Level: level method starts with the mean-value solution
	// option = 0: solving SAA for getting a candidate solution; option = 1: solving SAA for evaluating a given solution using CI
	// first solve a mean-value problem
	double accuracy;
	if (option == 0)
	{
		// For getting a candidate solution
		accuracy = 1e-6;
	}
	if (option == 1)
	{
		// For evaluating a given solution using CI
		accuracy = 1e-4;
	}
	double starttime = clock.getTime();
	IloEnv meanenv;
	IloNumArray meanxvals(meanenv);
	double meanobj = solve_mean_value_model(prob, meanenv, meanxvals, samples);
	vector<double> xiterate(prob.nbFirstVars);
	// Assign xiterate to meanx
	for (int j = 0; j < prob.nbFirstVars; ++j)
		xiterate[j] = meanxvals[j];
	bool feas_flag = 1;
	meanxvals.end();
	meanenv.end();

	// Define Master program
	IloModel mastermodel(env);
	IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
	// For now we assume an LB for the second-stage obj: 0
	IloNumVar theta(env, 0, IloInfinity);
	// Adding first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		mastermodel.add(range);
		lhs.end();
	}

	// Adding objective
	IloExpr obj(env);
	for (int i = 0; i < prob.nbFirstVars; ++i)
		obj += x[i]*prob.objcoef[i];
	obj += theta;
	mastermodel.add(IloMinimize(env, obj));
	obj.end();
	IloCplex mastercplex(mastermodel);
	mastercplex.setParam(IloCplex::TiLim,10800);
	mastercplex.setParam(IloCplex::Threads, 1);
	mastercplex.setParam(IloCplex::BarDisplay, 0);
	mastercplex.setParam(IloCplex::SimDisplay, 0);
	mastercplex.setOut(env.getNullStream());
	// Define Level quadratic program
	IloEnv lenv;
	IloModel levelmodel(lenv);
	IloNumVarArray lx(lenv, prob.firstvarlb, prob.firstvarub);
	IloNumVar ltheta(lenv, -IloInfinity, IloInfinity);
	// Adding first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(lenv);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += lx[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(lenv, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		levelmodel.add(range);
		lhs.end();
	}
	// define the first constraint for level master
	IloExpr lsum(lenv);
	for (int j = 0; j < prob.nbFirstVars; ++j)
		lsum += lx[j]*prob.objcoef[j];
	lsum += ltheta;
	IloRange rangeub(lenv, -IloInfinity, lsum, IloInfinity);
	levelmodel.add(rangeub);
	lsum.end();

	IloObjective lobj = IloMinimize(lenv);
	levelmodel.add(lobj);

	IloCplex levelcplex(levelmodel);
	levelcplex.setParam(IloCplex::TiLim,10800);
	levelcplex.setParam(IloCplex::Threads, 1);
	levelcplex.setParam(IloCplex::BarDisplay, 0);
	levelcplex.setOut(env.getNullStream());

	// We initiate an LP model for the second-stage problem, and everytime (iteration/sceanrio) we just update the rhs and solve: constraint coefficients are the same across all scenarios
	// we initiate both optimization model and feasibility model
	Subprob subp;
	construct_second_opt(env, prob, subp);
	construct_second_feas(env, prob, subp);

	int nbScens = samples.size();

	// This part needs to be changed into Level method iteration
	while ((stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) > accuracy || feas_flag == 0)
	{
		feas_flag = 1;
		IloEnv env2;
		stat.iter++;
		IloNumArray xiteratevals(env2, prob.nbFirstVars);
		for (int j = 0; j < prob.nbFirstVars; ++j)
		{
			xiterateXf(j) = xiterate[j];
			xiteratevals[j] = xiterate[j];
		}
		double feasbound = 0.0;
		for (int j = 0; j < prob.nbFirstVars; ++j)
			feasbound += xiterate[j]*prob.objcoef[j];

		IloExpr lhsaggr(env);
		lhsaggr += theta;
		IloExpr llhsaggr(lenv);
		llhsaggr += ltheta;
		double rhsaggr = 0;
		double lasttime = clock.getTime();
		for (int k = 0; k < nbScens; ++k)
		{
			// solve subproblems for each partition
			IloNumArray duals(env2);
			bool feasflag;
			double subobjval = subprob(subp, prob, xiteratevals, duals, samples[k], feasflag);
			VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
			for (int i = 0; i < prob.nbSecRows+prob.nbSecVars; ++i)
				dualvec(i) = duals[i];
			duals.end();
			if (feasflag == 1)
			{
				// optimal, so return extreme point solution
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
				double sum_xvals = opt_cut_coef.dot(xiterateXf);
				double rhssub = subobjval + sum_xvals;
				feasbound += subobjval*1.0/nbScens;
				rhsaggr += rhssub*1.0/nbScens;
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
					{
						lhsaggr += x[j]*opt_cut_coef[j]*1.0/nbScens;
						llhsaggr += lx[j]*opt_cut_coef[j]*1.0/nbScens;
					}
				}

			}
			else
			{
				feas_flag = 0;
				// infeasible, so return extreme rays
				VectorXf feas_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0, prob.nbSecRows);
				double sum_xvals = feas_cut_coef.dot(xiterateXf);
				IloExpr lhssub(env);
				IloExpr llhssub(lenv);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(feas_cut_coef[j]) > 1e-7)
					{
						llhssub += lx[j]*feas_cut_coef[j];
						lhssub += x[j]*feas_cut_coef[j];
					}
				}
				double rhssub = sum_xvals+subobjval;
				stat.num_feas_cuts++;
				mastermodel.add(lhssub >= rhssub);
				levelmodel.add(llhssub >= rhssub);
				lhssub.end();
				llhssub.end();
			}
		}
		xiteratevals.end();
		stat.subtime += clock.getTime()-lasttime;
		if (feas_flag == 1)
		{
			//cout << "feasbound = " << feasbound << ", feasobjval = " << stat.feasobjval << endl;
			if (feasbound <= stat.feasobjval)
				stat.feasobjval = feasbound;
			levelmodel.add(llhsaggr >= rhsaggr);
			mastermodel.add(lhsaggr >= rhsaggr);
			stat.num_opt_cuts++;
		}
		lhsaggr.end();
		llhsaggr.end();
		// Now solve the master, get a lower bound
		lasttime = clock.getTime();
		mastercplex.solve();
		stat.relaxobjval = mastercplex.getObjValue();
		stat.mastertime += clock.getTime()-lasttime;

		// Now solve the qp level problem
		// update the upper bound, (1-\lambda)F^{k+1}+\lambda F^*
		rangeub.setUB(0.5*stat.relaxobjval + 0.5*stat.feasobjval);
		IloExpr objExpr(lenv);
		for (int j = 0; j < prob.nbFirstVars; ++j)
		{
			objExpr += lx[j]*lx[j];
			objExpr -= lx[j]*2*xiterate[j];
		}
		lobj.setExpr(objExpr);
		objExpr.end();

		double startqptime = clock.getTime();
		levelcplex.solve();
		stat.qptime += clock.getTime()-startqptime;
		IloNumArray lxval(lenv);
		levelcplex.getValues(lxval, lx);
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterate[j] = lxval[j];
		lxval.end();
		env2.end();
		//cout << "relaxobjval = " << stat.relaxobjval << endl;
		//cout << "feasobjval = " << stat.feasobjval << endl;
		//cout << "optimality gap = " << (stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) << endl;
		stat.solvetime = clock.getTime()-starttime;
		if (stat.solvetime > 10800)
			break;
	}
	subp.suboptcplex.end();
	subp.suboptmodel.end();
	subp.suboptcon.end();
	subp.subopty.end();
	subp.subfeascplex.end();
	subp.subfeasmodel.end();
	subp.subfeascon.end();
	subp.subfeasy.end();
	mastercplex.end();
	mastermodel.end();
	levelcplex.end();
	levelmodel.end();
	lx.end();
	x.end();
}

bool addToCollection(const VectorXf& dualvec, vector<DualInfo>& dualInfoCollection)
{
	// try to add dualvec into dualCollection: if it is not parallel to any other vector in the collection
	bool flag = 1;
	for (int j = 0; j < dualInfoCollection.size(); ++j)
	{
		double par = dualvec.dot(dualInfoCollection[j].dualvec)*1.0/(dualvec.norm()*dualInfoCollection[j].dualvec.norm());
		if (par > 1-1e-3)
		{
			flag = 0;
			break;
		}
	}
	return flag;
}


void solve_adaptive(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, int option, bool saaError)
{
	// option = 0: B&M (2011)
	// option = 1,2: B&P-L FSP/SSP
	// option = 3: use fixed rate schedule on increasing sample size, when BM fails to progress - when the new sampled problem is nearly optimal at the very first iteration for >= 3 times; also, once we switch to fixed rate schedule, we keep it in that way!
	// option = 4: use fixed rate schedule all the way
	// option = 5: use a simple heuristic "trust region" idea to adjust the sample size increasing rate

	// saaError = 0: traditional sequential sampling
	// saaError = 1: adaptive: solve SAA up to the sampling error

	Sequence seq;
	sequentialSetup(seq, option);
	double epsilon; // epsilon is for stopping criterion of B&P-L
	bool iterflag = 1;
	int iter = 0;

	// Construct subproblems
	Subprob subp;
	construct_second_opt(env, prob, subp);
	construct_second_feas(env, prob, subp);

	// collection of dual multipliers
	vector<DualInfo> dualInfoCollection;
	VectorXf xiterateXf(prob.nbFirstVars);
	vector<VectorXf> rhsvecs;
	// Store all the rhs vectors
	for (int k = 0; k < prob.nbScens; ++k)
	{
		VectorXf rhsXf(prob.nbSecRows);
		for (int j = 0; j < prob.nbSecRows; ++j)
			rhsXf[j] = prob.secondconstrbd[j+k*prob.nbSecRows];
		rhsvecs.push_back(rhsXf);
	}
	int nearOptimal = 0; // keep track of # of consecutive times where the new sampled problem is nearly optimal at the very first iteration

	int nbIterEvalScens, nbIterSolScens;
	double starttime = clock.getTime();

	// We create some safeguard in case we do not finish by satisfying the stopping criterion in nMax steps
	double minGS = 1e8;
	double minGCI = 1e8;
	//double increaseRate = 0.1; // Starting from 0.1, lower bounded by 0.05
	while (iterflag == 1) // Outer loop
	{
		/* Setting up the sample size */
		if (iter == 0 || option == 0 || option == 1 || option == 2)
			nbIterEvalScens = seq.sampleSizes[iter];
		else
		{
			if (option == 4 || option == 5)
			{
				// option = 4: use a fixed rate all the way
				// option = 5: use a simple heuristic trust region idea to choose the sample size increasing rate
				nbIterEvalScens = int(nbIterEvalScens * (1+prob.increaseRate));
				if (nbIterEvalScens < seq.sampleSizes[iter])
					nbIterEvalScens = seq.sampleSizes[iter];
			}
			if (option == 3)
			{
				if (nearOptimal >= 3)
				{
					nbIterEvalScens = int(nbIterEvalScens * (1+prob.increaseRate));
					if (nbIterEvalScens < seq.sampleSizes[iter])
						nbIterEvalScens = seq.sampleSizes[iter];
				}
				else
				{
					if (seq.sampleSizes[iter] > nbIterEvalScens)
						nbIterEvalScens = seq.sampleSizes[iter];
				}
			}
		}
		nbIterSolScens = 2*nbIterEvalScens;
		cout << "nbIterEvalScens = " << nbIterEvalScens << ", nbIterSolScens = " << nbIterSolScens << endl;

		/* Begin Solving sampled problems */
		vector<int> samplesForSol;
		bool evalFlag = 1;
		if (nbIterSolScens >= prob.nbScens)
		{
			// In case nbIterSolScens gets too big!
			for (int j = 0; j < prob.nbScens; ++j)
				samplesForSol.push_back(j);
			// No need to do probabilistic evaluation
			evalFlag = 0;
		}
		else
		{
			samplesForSol = vector<int>(nbIterSolScens);
			for (int j = 0; j < nbIterSolScens; ++j)
				samplesForSol[j] = rand() % prob.nbScens;
		}
		/* Construct Master problem in each iteration */
		IloModel mastermodel(env);
		IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
		IloNumVarArray theta(env, nbIterSolScens, 0, IloInfinity);
		// Adding first-stage constraints
		for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
		{
			IloExpr lhs(env);
			for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
				lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
			IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
			mastermodel.add(range);
			lhs.end();
		}
		// Adding objective
		IloExpr obj(env);
		for (int i = 0; i < prob.nbFirstVars; ++i)
			obj += x[i]*prob.objcoef[i];
		for (int k = 0; k < nbIterSolScens; ++k)
			obj += theta[k]*(1.0/nbIterSolScens);
		mastermodel.add(IloMinimize(env, obj));
		obj.end();

		IloCplex mastercplex(mastermodel);
		mastercplex.setParam(IloCplex::TiLim,10800);
		mastercplex.setParam(IloCplex::Threads, 1);
		// Dual simplex
		mastercplex.setParam(IloCplex::RootAlg, 2);
		mastercplex.setParam(IloCplex::BarDisplay, 0);
		mastercplex.setParam(IloCplex::SimDisplay, 0);
		mastercplex.setOut(env.getNullStream());
		mastercplex.setParam(IloCplex::EpOpt, 1e-6);
		/* Finish construct Master problem in each iteration */

		// Adding all cutting planes from a collection of duals as constraints
		IloRangeArray cutcon(env);
		addInitialCuts(env, prob, mastermodel, x, theta, mastercplex, cutcon, samplesForSol, dualInfoCollection, xiterateXf, rhsvecs);
		bool newSampleFlag = 0;
		int innerIter = 0;
		double feasobjval = 1e10;
		double relaxobjval = -1e10;
		while (newSampleFlag == 0) // Inner loop
		{
			// After obtaining samples, solve the sampled problem, obtain xval, relaxation bound for the sampled problem
			innerIter++;
			if (innerIter > 1 || dualInfoCollection.size()*samplesForSol.size() < 10000)
				mastercplex.solve();
			relaxobjval = mastercplex.getObjValue();
			cout << "innerIter = " << innerIter << ", relaxobjval = " << relaxobjval << ", feasobjval = " << feasobjval << endl;;
			IloNumArray xvals(env);
			mastercplex.getValues(xvals, x);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				xiterateXf(j) = xvals[j];
			// solve the sampled subproblems, generate cuts, and then get an upper bound for the sampled problem
			double tempfeasobjval = 0;
			vector<double> scenobj;
			bool cutflag = 0;
			double sampleMean = 0;
			IloNumArray scenthetaval(env);
			mastercplex.getValues(scenthetaval, theta);
			double tempsubprobtime = clock.getTime();
			bool feas_flag = 1;
			for (int kk = 0; kk < nbIterSolScens; ++kk)
			{
				int k = samplesForSol[kk];
				IloNumArray duals(env);
				bool feasflag;
				double subobjval = subprob(subp, prob, xvals, duals, k, feasflag);
				VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
				for (int i = 0; i < prob.nbSecRows+prob.nbSecVars; ++i)
					dualvec(i) = duals[i];
				duals.end();
				if (feasflag == 1)
				{
					scenobj.push_back(subobjval);
					sampleMean += subobjval;
					if (subobjval > scenthetaval[kk] + max(1e-5, abs(scenthetaval[kk])) * 1e-5)
					{
						VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
						double sum_xvals = 0;
						for (int j = 0; j < prob.nbFirstVars; ++j)
						{
							if (fabs(opt_cut_coef[j]) > 1e-7)
								sum_xvals += opt_cut_coef[j]*xvals[j];
						}
						double rhssub = subobjval + sum_xvals;
						// Need to add cuts here!
						IloExpr lhs(env);
						lhs += theta[kk];
						for (int j = 0; j < prob.nbFirstVars; ++j)
						{
							if (fabs(opt_cut_coef[j]) > 1e-7)
								lhs += x[j]*opt_cut_coef[j];
						}
						mastermodel.add(lhs >= rhssub);
						lhs.end();
						if (addToCollection(dualvec, dualInfoCollection) == true)
						{
							cutflag = 1;
							DualInfo dual;
							dual.dualvec = dualvec;
							dual.rhs = rhssub - dualvec.segment(0,prob.nbSecRows).transpose()*rhsvecs[k];
							dual.coefvec = opt_cut_coef;
							dualInfoCollection.push_back(dual);
						}
					}
					tempfeasobjval += subobjval;
				}
				else
				{
					cout << "; feas cut! Infeasibility = " << subobjval << endl;
					cutflag = 1;
					feas_flag = 0;
					VectorXf feas_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0, prob.nbSecRows);
					double sum_xvals = feas_cut_coef.dot(xiterateXf);
					IloExpr lhssub(env);
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(feas_cut_coef[j]) > 1e-7)
							lhssub += x[j]*feas_cut_coef[j];
					}
					double rhssub = sum_xvals+subobjval;
					stat.num_feas_cuts++;
					mastermodel.add(lhssub >= rhssub);
					lhssub.end();
					// adding as first-stage constraints to be used in the future
					IloIntArray tempInd(env);
					IloNumArray tempCoef(env);
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(feas_cut_coef[j]) > 1e-7)
						{
							tempInd.add(j);
							tempCoef.add(feas_cut_coef[j]);
						}
					}
					prob.firstconstrind.add(tempInd);
					prob.firstconstrcoef.add(tempCoef);
					prob.firstconstrlb.add(rhssub);
					prob.firstconstrub.add(IloInfinity);
					break;
				}
			}
			scenthetaval.end();
			if (feas_flag == 1)
			{
				// The following will only happen if no feasibility cuts are generated
				tempfeasobjval = tempfeasobjval*1.0/nbIterSolScens;
				for (int j = 0; j < prob.nbFirstVars; ++j)
					tempfeasobjval += xvals[j]*prob.objcoef[j];
				if (tempfeasobjval < feasobjval)
					feasobjval = tempfeasobjval;
				// Get the optimality gap with respective to the current sample
				double gap = feasobjval - relaxobjval;
				/* Now decide if we should solve a new sampled problem (with a larger sample) or continue solving the current sampled problem */
				if (saaError == 1 && evalFlag == 1)
				{
					// Get the sampling error
					double samplingError = 0;
					sampleMean = sampleMean*1.0/nbIterSolScens;
					for (int kk = 0; kk < nbIterSolScens; ++kk)
						samplingError += pow(scenobj[kk]-sampleMean, 2);
					samplingError = sqrt(samplingError)*1.0/nbIterSolScens;
					cout << "gap = " << gap << ", sample Error = " << samplingError << ", cutflag = " << cutflag << endl;
					if (samplingError > gap*10 || cutflag == 0)
						newSampleFlag = 1;
				}
				else
				{
					cout << "gap = " << gap << ", cutflag = " << cutflag << endl;
					if (gap*1.0/(fabs(feasobjval)+1e-10) <= 1e-6 || cutflag == 0)
						newSampleFlag = 1;
				}
			}
			if (newSampleFlag == 1)
			{
				if (evalFlag == 0)
				{
					iterflag = 0;
					stat.solvetime = clock.getTime()-starttime;
					stat.mainIter = iter;
					stat.finalSolExactObj = feasobjval;
					stat.gapThreshold = 0;
					stat.finalSampleSize = prob.nbScens;
					cout << "final objective = " << stat.finalSolExactObj << ", calculated by the full set of scenarios!" << endl;
				}
				else
				{
					// Finish inner iteration, need to do some SRP type testing
					if (innerIter == 1) //nearly optimal after the very first iteration
					{
						if (nearOptimal == 0)
							nearOptimal = 1;
						else
							nearOptimal++;
					}
					else
					{
						if (nearOptimal < 3)
							nearOptimal = 0;
					}
					// Now obtained xiterateXf, xvals, scenobj corresponding to \hat{x}_k
					if ((option == 1 || option == 2) && iter == 0)
					{
						// epsilon is for stopping criterion of B&P-L, just set it to be small enough relative to the initial UB obtained from the first iteration
						epsilon = prob.eps*fabs(feasobjval);
					}
					// Test stopping criterion: SRP type CI estimation
					double tempEvaltime = clock.getTime();
					double G = 0;
					double S = 0;
					vector<double> scenObjEval(nbIterEvalScens);
					vector<int> samplesForEval(nbIterEvalScens);
					for (int j = 0; j < nbIterEvalScens; ++j)
						samplesForEval[j] = rand() % prob.nbScens;
					STAT tempstat;
					tempstat.relaxobjval = -1e10;
					tempstat.feasobjval = 1e10;
					VectorXf xiterateXf2(prob.nbFirstVars);
					solve_level(env, prob, tempstat, clock, samplesForEval, xiterateXf2, 1);
					IloNumArray xvals2(env, prob.nbFirstVars);
					for (int j = 0; j < prob.nbFirstVars; ++j)
						xvals2[j] = xiterateXf2(j);
					// xiterateXf, xvals, scenObj correspondiing to \hat{x}_k, xiterateXf2, xvals2, scenObj2 corresponding to x^*_{n_k}
					SRP(env, prob, subp, clock, nbIterEvalScens, samplesForEval, G, S, xvals, xvals2, scenObjEval);
					xvals2.end();
					stat.evaltime += (clock.getTime()-tempEvaltime);

					// Check if the stopping criterion is met
					if (option == 0)
					{
						// B&M (2011)
						cout << "G/S = " << G*1.0/S << ", G = " << G << ", S = " << S << endl;
						if (G*1.0/S < minGS)
							minGS = G*1.0/S;
						if (G <= seq.h2*S + seq.eps2)
							iterflag = 0;
						else
							iter++;
					}
					if (option == 1 || option == 2)
					{
						cout << "CI width = " << G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) << ", epsilon = " << epsilon << endl;
						// B&P-L: FSP/SSP
						if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) < minGCI)
							minGCI = G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens);
						if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) <= epsilon)
							iterflag = 0;
						else
						{
							iter++;
							if (option == 2)
							{
								// sample size for the next iteration will depend (adaptively) on the statistics of the current iteration
								double const_b = t_quant*S+1;
								double const_c = nbIterEvalScens*G;
								double const_delta = const_b*const_b + 4*epsilon*const_c;
								double const_v = (const_b + sqrt(const_delta))*1.0/(2*epsilon);
								seq.sampleSizes[iter] = int(const_v*const_v)+1;
							}
						}
					}
					if (option == 3)
					{
						// Start with BM - if BM fails to progress - switch to fixed (exponential) rate schedule
						// Use either B&M (2011) or B&P-L criteria
						if (G*1.0/S < minGS)
							minGS = G*1.0/S;
						if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) < minGCI)
							minGCI = G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens);
						if (G <= seq.h2*S + seq.eps2 || G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) <= epsilon)
							iterflag = 0;
						else
							iter++;
					}
					if (iter == seq.nMax)
						iterflag = 0;
					if (iterflag == 0)
					{
						cout << "# of iterations = " << iter << endl;
						stat.solvetime = clock.getTime()-starttime;

						// Now check if this estimation is correct by evaluating \hat{x}_k using the true distribution (all samples)
						finalEval(env, prob, subp, xiterateXf, stat);
						stat.mainIter = iter;
						if (option == 0)
						{
							// B&M (2011)
							if (iter != seq.nMax)
								stat.gapThreshold = seq.h*S+seq.eps;
							else
							{
								stat.gapThreshold = (minGS+seq.h-seq.h2)*S+seq.eps;
							}
							cout << "optimality gap is less than " << stat.gapThreshold << " with prob >= 90%" << endl;
						}
						if (option == 1 || option == 2)
						{
							// B&P-L: FSP or SSP
							if (iter != seq.nMax)
								stat.gapThreshold = G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens);
							else
							{
								stat.gapThreshold = minGCI;
							}
							cout << "optimality gap is less than " << stat.gapThreshold << " with prob >= 90%" << endl;
						}
						if (option == 3 || option == 4 || option == 5)
						{
							// Use either B&M (2011) or B&P-L criteria
							if (iter != seq.nMax)
							{
								double gapThreshold = 1e8;
								if (G <= seq.h2*S + seq.eps2)
									gapThreshold = seq.h*S + seq.eps;
								if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) <= epsilon && G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) < gapThreshold)
									gapThreshold = G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens);
								stat.gapThreshold = gapThreshold;
							}
							else
							{
								if ((minGS+seq.h-seq.h2)*S+seq.eps < minGCI)
									stat.gapThreshold = (minGS+seq.h-seq.h2)*S+seq.eps;
								else
									stat.gapThreshold = minGCI;
							}
							cout << "optimality gap is less than " << stat.gapThreshold << " with prob >= 90%" << endl;
						}
						stat.finalSampleSize = nbIterSolScens;
					}
				}
			}
			xvals.end();
		}
		stat.iter += innerIter;
		cutcon.endElements();
		cutcon.end();
		mastercplex.end();
		mastermodel.end();
		x.end();
		theta.end();
	}
	subp.suboptcplex.end();
	subp.suboptmodel.end();
	subp.suboptcon.end();
	subp.subopty.end();
	subp.subfeascplex.end();
	subp.subfeasmodel.end();
	subp.subfeascon.end();
	subp.subfeasy.end();
}

void addInitialCuts(IloEnv& env, TSLP& prob, IloModel& mastermodel, const IloNumVarArray& x, const IloNumVarArray& theta, IloCplex& mastercplex, IloRangeArray& cutcon, const vector<int>& samplesForSol, const vector<DualInfo>& dualInfoCollection, const VectorXf& xiterateXf, const vector<VectorXf>& rhsvecs)
{
	// Given a collection of dual multipliers, construct an initial master problem (relaxation)
	if (dualInfoCollection.size()*samplesForSol.size() < 10000)
	{
		// Number of constraints is small enough to handle
		for (int l = 0; l < dualInfoCollection.size(); ++l)
		{
			for (int kk = 0; kk < samplesForSol.size(); ++kk)
			{
				int k = samplesForSol[kk];
				// assemble the cutcoef and cutrhs
				VectorXf opt_cut_coef = dualInfoCollection[l].coefvec;
				double opt_cut_rhs = dualInfoCollection[l].dualvec.segment(0,prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[l].rhs;
				IloExpr lhs(env);
				lhs += theta[kk];
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) > 1e-7)
						lhs += x[j]*opt_cut_coef[j];
				}
				IloRange range(env, opt_cut_rhs, lhs, IloInfinity);
				mastermodel.add(range);
				cutcon.add(range);
				lhs.end();
			}
		}
	}
	else
	{
		// Too many initial constraints, add them as cutting planes
		for (int kk = 0; kk < samplesForSol.size(); ++kk)
		{
			int k = samplesForSol[kk];
			// assemble the cutcoef and cutrhs
			double maxval = -1e8;
			int maxind = -1;
			for (int l = 0; l < dualInfoCollection.size(); ++l)
			{
				VectorXf opt_cut_coef = dualInfoCollection[l].coefvec;
				double opt_cut_rhs = dualInfoCollection[l].dualvec.segment(0,prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[l].rhs;
				double rhsval = opt_cut_rhs - opt_cut_coef.transpose() * xiterateXf;
				if (rhsval > maxval)
				{
					maxval = rhsval;
					maxind = l;
				}
			}
			IloExpr lhs(env);
			lhs += theta[kk];
			VectorXf init_cut_coef = dualInfoCollection[maxind].coefvec;
			for (int j = 0; j < prob.nbFirstVars; ++j)
			{
				if (fabs(init_cut_coef[j]) > 1e-7)
					lhs += x[j]*init_cut_coef[j];
			}
			IloRange range(env, dualInfoCollection[maxind].dualvec.segment(0,prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[maxind].rhs, lhs, IloInfinity);
			mastermodel.add(range);
			cutcon.add(range);
			lhs.end();
		}
		bool initialCutFlag = 1;
		while (initialCutFlag == 1)
		{
			initialCutFlag = 0;
			mastercplex.solve();
			IloNumArray xvals(env);
			IloNumArray thetavals(env);
			mastercplex.getValues(xvals, x);
			mastercplex.getValues(thetavals, theta);
			VectorXf tempxiterateXf(prob.nbFirstVars);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				tempxiterateXf(j) = xvals[j];
			for (int kk = 0; kk < samplesForSol.size(); ++kk)
			{
				int k = samplesForSol[kk];
				// assemble the cutcoef and cutrhs
				double maxval = -1e8;
				int maxind = -1;
				for (int l = 0; l < dualInfoCollection.size(); ++l)
				{
					VectorXf opt_cut_coef = dualInfoCollection[l].coefvec;
					double opt_cut_rhs = dualInfoCollection[l].dualvec.segment(0,prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[l].rhs;
					double rhsval = opt_cut_rhs - opt_cut_coef.transpose() * tempxiterateXf - thetavals[kk];
					if (rhsval > maxval)
					{
						maxval = rhsval;
						maxind = l;
					}
				}
				if (maxval > max(1e-5, abs(thetavals[kk])) * 1e-5)
				{
					initialCutFlag = 1;
					IloExpr lhs(env);
					lhs += theta[kk];
					VectorXf init_cut_coef = dualInfoCollection[maxind].coefvec;
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(init_cut_coef[j]) > 1e-7)
							lhs += x[j]*init_cut_coef[j];
					}
					IloRange range(env, dualInfoCollection[maxind].dualvec.segment(0,prob.nbSecRows).transpose() * rhsvecs[k] + dualInfoCollection[maxind].rhs, lhs, IloInfinity);
					mastermodel.add(range);
					cutcon.add(range);
					lhs.end();
				}
			}
			xvals.end();
			thetavals.end();
		}
	}
}

void construct_second_opt(IloEnv& env, const TSLP& prob, Subprob& subprob)
{
	subprob.suboptmodel = IloModel(env);
	subprob.suboptcon = IloRangeArray(env);
	subprob.subopty = IloNumVarArray(env, prob.nbSecVars, -IloInfinity, IloInfinity);
	subprob.suboptcplex = IloCplex(subprob.suboptmodel);
	subprob.suboptcplex.setParam(IloCplex::TiLim, 3600);
	subprob.suboptcplex.setParam(IloCplex::Threads, 1);
	subprob.suboptcplex.setParam(IloCplex::SimDisplay, 0);
	subprob.suboptcplex.setParam(IloCplex::BarDisplay, 0);
	subprob.suboptcplex.setParam(IloCplex::EpRHS, 5e-6);
	subprob.suboptcplex.setOut(env.getNullStream());
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs += subprob.subopty[ind-prob.nbFirstVars]*prob.CoefMat[i][j];
		}
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.suboptcon.add(range);
		subprob.suboptmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs += subprob.subopty[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.suboptcon.add(range);
		subprob.suboptmodel.add(range);
		lhs.end();
	}
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs -= subprob.subopty[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.suboptcon.add(range);
		subprob.suboptmodel.add(range);
		lhs.end();
	}
	// second-stage obj
	IloExpr suboptobj(env);
	for (int i = 0; i < prob.objcoef.getSize(); ++i)
	{
		if (i >= prob.nbFirstVars)
			suboptobj += subprob.subopty[i-prob.nbFirstVars]*prob.objcoef[i];
	}
	subprob.suboptmodel.add(IloMinimize(env, suboptobj));
	suboptobj.end();
}

void construct_second_feas(IloEnv& env, const TSLP& prob, Subprob& subprob)
{
	subprob.subfeasmodel = IloModel(env);
	subprob.subfeascon = IloRangeArray(env);
	subprob.subfeasy = IloNumVarArray(env, prob.nbSecVars+prob.nbSecRows, -IloInfinity, IloInfinity);
	subprob.subfeascplex = IloCplex(subprob.subfeasmodel);
	subprob.subfeascplex.setParam(IloCplex::TiLim, 3600);
	subprob.subfeascplex.setParam(IloCplex::Threads, 1);
	subprob.subfeascplex.setParam(IloCplex::SimDisplay, 0);
	subprob.subfeascplex.setParam(IloCplex::BarDisplay, 0);
	subprob.subfeascplex.setOut(env.getNullStream());
	vector<int> extra_ind(prob.nbSecRows, -1);
	for (int j = 0; j < prob.nbSecRows; ++j)
	{
		if (prob.secondconstrsense[j] == -1)
			subprob.subfeasy[prob.nbSecVars+j].setLB(0);
		if (prob.secondconstrsense[j] == 0)
		{
			subprob.subfeasy[prob.nbSecVars+j].setLB(0);
			IloNumVar temp(env, 0, IloInfinity);
			subprob.subfeasy.add(temp);
			extra_ind[j] = subprob.subfeasy.getSize()-1;
		}
		if (prob.secondconstrsense[j] == 1)
			subprob.subfeasy[prob.nbSecVars+j].setUB(0);
	}
	// second-stage constraints
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.nbPerRow[i]; ++j)
		{
			int ind = prob.CoefInd[i][j];
			if (ind >= prob.nbFirstVars)
				lhs += subprob.subfeasy[ind-prob.nbFirstVars]*prob.CoefMat[i][j];
		}
		if (prob.secondconstrsense[i] != 0)
			lhs += subprob.subfeasy[prob.nbSecVars+i];
		else
		{
			lhs += subprob.subfeasy[prob.nbSecVars+i];
			lhs -= subprob.subfeasy[extra_ind[i]];
		}
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.subfeascon.add(range);
		subprob.subfeasmodel.add(range);
		lhs.end();
	}
	// second-stage variable bounds
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs += subprob.subfeasy[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.subfeascon.add(range);
		subprob.subfeasmodel.add(range);
		lhs.end();
	}
	for (int i = 0; i < prob.nbSecVars; ++i)
	{
		IloExpr lhs(env);
		lhs -= subprob.subfeasy[i];
		IloRange range(env, -IloInfinity, lhs, IloInfinity);
		subprob.subfeascon.add(range);
		subprob.subfeasmodel.add(range);
		lhs.end();
	}
	// second-stage obj
	IloExpr subfeasobj(env);
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		if (prob.secondconstrsense[i] == -1)
			subfeasobj += subprob.subfeasy[prob.nbSecVars+i];
		if (prob.secondconstrsense[i] == 1)
			subfeasobj -= subprob.subfeasy[prob.nbSecVars+i];
		if (prob.secondconstrsense[i] == 0)
		{
			subfeasobj += subprob.subfeasy[prob.nbSecVars+i];
			subfeasobj += subprob.subfeasy[extra_ind[i]];
		}
	}
	subprob.subfeasmodel.add(IloMinimize(env, subfeasobj));
	subfeasobj.end();
}

void simple_refine(const Component& component, const TSLP& prob, const vector<IloNumArray>& extreme_points, const vector<int>& extreme_points_ind, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, vector<Component>& new_partition, vector< vector<int> >& extreme_ray_map)
{
	// Simple refinement strategy by putting vectors together according to a distance threshold
	vector<int> index_points_represent, index_rays_represent; // list of representing indices
	vector< vector<int> > index_points, index_rays; // index globally defined
	for (int i = 0; i < extreme_points.size(); ++i)
	{
		if (i == 0)
		{
			vector<int> newlist;
			newlist.push_back(extreme_points_ind[i]);
			index_points_represent.push_back(i);
			index_points.push_back(newlist);
		}
		else
		{
			bool distinctflag = 1;
			for (int j = 0; j < index_points_represent.size(); ++j)
			{
				bool tempflag = compare_arrays(prob, extreme_points[index_points_represent[j]], extreme_points[i]);
				if (tempflag == 1)
				{
					index_points[j].push_back(extreme_points_ind[i]);
					distinctflag = 0;
					break;
				}
			}
			if (distinctflag == 1)
			{
				vector<int> newlist;
				newlist.push_back(extreme_points_ind[i]);
				index_points_represent.push_back(i);
				index_points.push_back(newlist);
			}
		}
	}

	for (int i = 0; i < extreme_rays.size(); ++i)
	{
		if (i == 0)
		{
			// Since this is the first one, create a group starting with this one
			vector<int> newlist;
			newlist.push_back(extreme_rays_ind[i]);
			index_rays_represent.push_back(i);
			index_rays.push_back(newlist);
			vector<int> temp;
			temp.push_back(i);
			extreme_ray_map.push_back(temp);
		}
		else
		{
			bool distinctflag = 1;
			for (int j = 0; j < index_rays_represent.size(); ++j)
			{
				bool tempflag = compare_arrays(prob, extreme_rays[index_rays_represent[j]], extreme_rays[i]);
				if (tempflag == 1)
				{
					index_rays[j].push_back(extreme_rays_ind[i]);
					extreme_ray_map[j].push_back(i);
					distinctflag = 0;
					break;
				}
			}
			if (distinctflag == 1)
			{
				vector<int> newlist;
				newlist.push_back(extreme_rays_ind[i]);
				index_rays_represent.push_back(i);
				index_rays.push_back(newlist);
				vector<int> temp;
				temp.push_back(i);
				extreme_ray_map.push_back(temp);
			}
		}
	}
	for (int j = 0; j < index_rays.size(); ++j)
	{
		Component compo;
		compo.indices = index_rays[j];
		new_partition.push_back(compo);
	}
	for (int j = 0; j < index_points.size(); ++j)
	{
		Component compo;
		compo.indices = index_points[j];
		new_partition.push_back(compo);
	}
}

bool compare_arrays(const TSLP& prob, const IloNumArray& array1, const IloNumArray& array2)
{
	// check if they are "equal" to each other
	bool returnflag = 1;
	for (int j = 0; j < array1.getSize(); ++j)
	{
		if (fabs((array1[j]-array2[j])*1.0/(array1[j]+1e-5)) > prob.distinct_par)
		{
			returnflag = 0;
			break;
		}
	}
	return returnflag;
}

bool solve_partly_inexact_bundle(IloEnv& env, TSLP& prob, Subprob& subp, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option, bool initial, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int& nearOptimal, double remaintime)
{
	// option = 0: Just use the relative opt gap, solution mode: 1e-6; option = 1: Use the adaptive sampling type threshold: when opt gap is small relative to the sample error, option = 2: use relative opt gap, evaluation mode: 1e-4
	// initial = 0: a stability center is provided as xiterteXf from the previous iteration; initial = 1: the very first sampled problem solved
	// Unified coarse/fine oracle as a partly inexact oracle
	bool returnflag = 1;
	if (option == 1)
		cout << "begin solution mode...up to sample error" << endl;
	if (option == 2)
		cout << "begin evaluation mode..." << endl;
	double eval_threshold = 1e-4;
	double starttime = clock.getTime();
	IloEnv env2;
	IloNumArray xvals(env2);
	double fmean;
	vector<Component> partition;
	bool feas_flag = 0;
	IloCplex cplex;
	IloModel model;
	IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
	IloObjective QPobj;
	IloNumArray stab_center(env, prob.nbFirstVars);
	vector< vector<double> > cutcoefs; // We may do some cut screening later
	vector<double> cutrhs;
	IloRangeArray cuts(env);
	IloEnv meanenv;
	double warmtemp = clock.getTime();
	if (initial == 0)
	{
		// One of the later iterations
		for (int j = 0; j < prob.nbFirstVars; ++j)
			stab_center[j] = xiterateXf(j);
		// We will also do some warm starts here using recorded dual information from dualCollection
		fmean = solve_warmstart(meanenv, prob, samples, stab_center, dualInfoCollection, cutcoefs, cutrhs, partition, rhsvecs, xvals, clock, stat);
		// Add cuts
		for (int l = 0; l < cutcoefs.size(); ++l)
		{
			IloExpr lhs(env);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				lhs += x[j]*(prob.objcoef[j]-cutcoefs[l][j]*1.0/samples.size());
			// Just tempararily set an UB, will be updated in the inner loop any way
			IloRange range(env, -IloInfinity, lhs, 0);
			cuts.add(range);
			lhs.end();
		}
		// This is only temporary
		Component all;
		for (int i = 0; i < samples.size(); ++i)
			all.indices.push_back(samples[i]);
		partition.push_back(all);
	}
	else
	{
		// Solve the mean-value problem first
		fmean = solve_mean_value_model(prob, meanenv, xvals, samples);
		// Set stab_center to be 0 initially
		for (int j = 0; j < prob.nbFirstVars; ++j)
			stab_center[j] = 0;
		// Initial partition: everybody is together, i.e., the mean-value problem
		Component all;
		for (int i = 0; i < samples.size(); ++i)
			all.indices.push_back(samples[i]);
		partition.push_back(all);
	}
	stat.warmstarttime = clock.getTime()-warmtemp;
	meanenv.end();
	IloRangeArray center_cons(env);
	setup_bundle_QP(env, prob, cplex, model, x, stab_center, QPobj, cuts, center_cons);
	bool firstloop = 1;
	double descent_target;
	double opt_gap;
	double samplingError = 0; // samplingError will be updated whenever there is a chance to loop through all scenarios
	bool loopflag = 1;
	while (loopflag == 1 || feas_flag == 0)
	{
		feas_flag = 1;
		stat.iter++;
		//cout << "stat.iter = " << stat.iter << endl;
		stat.partitionsize += partition.size();
		double feasboundscen = 0.0;
		VectorXf cutcoefscen(prob.nbFirstVars);
		cutcoefscen.setZero();
		VectorXf aggrCoarseCut(prob.nbFirstVars);
		vector<VectorXf> partcoef(partition.size());
		vector<double> partrhs(partition.size());
		double coarseLB, coarseCutRhs;
		vector<double> scenObjs; // record scenario objectives
		if (firstloop == 1)
			stat.relaxobjval = fmean;
		else
		{
			double lasttime = clock.getTime();
			coarseLB = coarse_oracle(env, prob, subp, partition, xvals, feasboundscen, cutcoefscen, cplex, model, x, stat, center_cons, stab_center, cuts, cutrhs, aggrCoarseCut, coarseCutRhs, partcoef, partrhs, starttime, clock, scenObjs, samples, dualInfoCollection, rhsvecs, option);
			stat.solvetime = clock.getTime()-starttime;
			if (stat.solvetime > remaintime)
			{
				returnflag = 0;
				break;
			}
			descent_target = prob.kappaf*stat.relaxobjval+(1-prob.kappaf)*stat.feasobjval;
			// Here opt_gap is updated since stat.relaxobjval may have been updated during coarse_oracle()
			opt_gap = stat.feasobjval-stat.relaxobjval;
			if (option == 1 && samplingError > opt_gap*10)
			{
				loopflag = 0;
				continue;
			}
			if (option == 0 && opt_gap*1.0/(fabs(stat.feasobjval)+1e-10) <= prob.eps)
			{
				loopflag = 0;
				continue;
			}
			if (option == 2 && opt_gap*1.0/(fabs(stat.feasobjval)+1e-10) <= eval_threshold)
			{
				//cout << "opt_gap = " << opt_gap << ", feasobjval = " << stat.feasobjval << endl;
				loopflag = 0;
				continue;
			}
			if (coarseLB > descent_target)
			{
				// Cannot achieve descent target, so just use coarse oracle
				IloExpr lhs(env);
				vector<double> tempcut(prob.nbFirstVars);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(aggrCoarseCut[j]) > 1e-7)
					{
						lhs += x[j]*(prob.objcoef[j]-aggrCoarseCut[j]*1.0/samples.size());
						tempcut[j] = aggrCoarseCut[j];
					}
					else
					{
						lhs += x[j]*prob.objcoef[j];
						tempcut[j] = 0;
					}
				}
				// Just tempararily set an UB, will be updated in the inner loop any way
				IloRange range(env, -IloInfinity, lhs, -coarseCutRhs);
				model.add(range);
				cuts.add(range);
				cutcoefs.push_back(tempcut);
				cutrhs.push_back(coarseCutRhs);
				lhs.end();
				stat.num_opt_cuts++;
				if (loopflag != 0)
					continue;
			}
		}
		// Begin partition refinement
		vector<Component> new_partition;
		bool fullupdateflag = 0;
		if (firstloop == 1)
		{
			// just solve all the scenario subproblems
			feas_flag = solve_scen_subprobs(env, env2, prob, subp, partition, xvals, feasboundscen, cutcoefscen, model, x, new_partition, stat, clock, scenObjs, dualInfoCollection, rhsvecs, option);
			fullupdateflag = 1;
		}
		else
		{
			// Solve scenario subproblems component by component, stop when hopeless to achieve descent target
			feas_flag = solve_scen_subprobs_target(env, env2, prob, subp, partition, xvals, model, x, new_partition, stat, clock, partcoef, partrhs, descent_target, fullupdateflag, coarseLB, aggrCoarseCut, scenObjs, samples, dualInfoCollection, rhsvecs, option);
		}
		if (feas_flag == 1 && fullupdateflag == 1)
		{
			if (scenObjs.size() != samples.size())
			{
				cout << "scenObjs.size != samples.size" << endl;
				exit(0);
			}
			// A new candidate is available, update UB
			if (firstloop == 1)
			{
				double feasbound = 0.0;
				for (int j = 0; j < prob.nbFirstVars; ++j)
					feasbound += xvals[j]*prob.objcoef[j];
				feasbound += feasboundscen*1.0/samples.size();
				if (feasbound < stat.feasobjval-1e-5)
					stat.feasobjval = feasbound;
				// update stablization center
				for (int j = 0; j < prob.nbFirstVars; ++j)
					stab_center[j] = xvals[j];
				computeSamplingError(samplingError, scenObjs);
			}
			else
			{
				if (coarseLB < stat.feasobjval-1e-5)
					stat.feasobjval = coarseLB;
				// update stablization center
				if (coarseLB < descent_target)
				{
					computeSamplingError(samplingError, scenObjs);
					for (int j = 0; j < prob.nbFirstVars; ++j)
						stab_center[j] = xvals[j];
					// Begin bundle compression
					if (cuts.getSize() >= 100)
					{
						// Only do bundle compression if maintaining too many cuts in the bundle
						double maxval = -1e5;
						vector<double> cutvals(cuts.getSize());
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							double cutval = 0;
							for (int jj = 0; jj < prob.nbFirstVars; ++jj)
							{
								if (fabs(cutcoefs[j][jj]) > 1e-7 && fabs(stab_center[jj]) > 1e-7)
									cutval -= cutcoefs[j][jj]*stab_center[jj];
							}
							cutval = cutval*1.0/samples.size();
							cutval += cutrhs[j];
							cutvals[j] = cutval;
							if (cutval > maxval)
								maxval = cutval;
						}
						vector<bool> cutactive(cuts.getSize(), 0);
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							if (maxval-cutvals[j] < 1e-5)
								cutactive[j] = 1;
							else
							{
								// remove the cut
								model.remove(cuts[j]);
								cuts[j].end();
							}
						}
						IloRangeArray newcuts(env);
						vector< vector<double> > newcutcoefs;
						vector<double> newcutrhs;
						for (int j = 0; j < cuts.getSize(); ++j)
						{
							if (cutactive[j] == 1)
							{
								newcuts.add(cuts[j]);
								newcutcoefs.push_back(cutcoefs[j]);
								newcutrhs.push_back(cutrhs[j]);
							}
						}
						cutcoefs = newcutcoefs;
						cuts = newcuts;
						cutrhs = newcutrhs;
					}
				}
			}
		}
		opt_gap = stat.feasobjval-stat.relaxobjval;
		//cout << "stat.iter = "<< stat.iter << ", relaxobjval = " << stat.relaxobjval << ", feasobjval = " << stat.feasobjval << ", optimality gap = " << opt_gap << ", samplingError = " << samplingError << endl;
		if (option == 0 && opt_gap*1.0/(fabs(stat.feasobjval)+1e-10) <= 1e-6)
			loopflag = 0;
		if (option == 1 && samplingError > opt_gap*10)
			loopflag = 0;
		if (option == 2 && opt_gap*1.0/(fabs(stat.feasobjval)+1e-10) <= eval_threshold)
			loopflag = 0;
		//if (partition.size() == new_partition.size() && feas_flag == 1)
		//	loopflag = 0;
		//cout << "After updates, loopflag = " << loopflag << ", feas_flag = " << feas_flag << ", partition.size = " << partition.size() << ", new_partition.size = " << new_partition.size() << endl;
		if (loopflag == 1 && firstloop == 0)
		{
			partition = new_partition;
			// Need to add an aggregated cut, if feasible: maybe a fine oracle, or a mixed fine/coarse oracle
			if (feas_flag == 1)
			{
				if (firstloop == 1)
				{
					double lhsval = 0;
					vector<double> tempcut(prob.nbFirstVars);
					IloExpr lhs(env);
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(cutcoefscen[j]) > 1e-7)
						{
							lhs += x[j]*(-cutcoefscen[j]*1.0/samples.size()+prob.objcoef[j]);
							lhsval += xvals[j]*cutcoefscen[j];
							tempcut[j] = cutcoefscen[j];
						}
						else
						{
							tempcut[j] = 0;
							if (fabs(prob.objcoef[j]) > 1e-7)
								lhs += x[j]*prob.objcoef[j];
						}
					}
					double fineCutRhs = (feasboundscen+lhsval)*1.0/samples.size();
					// Just tempararily set an UB, will be updated in the inner loop any way
					IloRange range(env, -IloInfinity, lhs, -fineCutRhs);
					model.add(range);
					cuts.add(range);
					cutcoefs.push_back(tempcut);
					cutrhs.push_back(fineCutRhs);
					stat.num_opt_cuts++;
					lhs.end();
				}
				else
				{
					coarseCutRhs = coarseLB;
					for (int j = 0; j < prob.nbFirstVars; ++j)
						coarseCutRhs -= prob.objcoef[j]*xvals[j];
					IloExpr lhs(env);
					vector<double> tempcut(prob.nbFirstVars);
					double tempval = 0;
					for (int j = 0; j < prob.nbFirstVars; ++j)
					{
						if (fabs(aggrCoarseCut[j]) > 1e-7)
						{
							lhs += x[j]*(-aggrCoarseCut[j]*1.0/samples.size()+prob.objcoef[j]);
							tempval += xvals[j]*aggrCoarseCut[j];
							tempcut[j] = aggrCoarseCut[j];
						}
						else
						{
							tempcut[j] = 0;
							lhs += x[j]*prob.objcoef[j];
						}
					}
					cutcoefs.push_back(tempcut);
					coarseCutRhs += tempval*1.0/samples.size();
					// Just tempararily set an UB, will be updated in the inner loop any way
					IloRange range(env, -IloInfinity, lhs, -coarseCutRhs);
					model.add(range);
					cuts.add(range);
					cutrhs.push_back(coarseCutRhs);
					stat.num_opt_cuts++;
					lhs.end();
				}
			}
		}
		firstloop = 0;
		stat.solvetime = clock.getTime()-starttime;
		if (stat.solvetime > remaintime)
		{
			returnflag = 0;
			break;
		}
	}
	if (stat.iter == 1) //nearly optimal after the very first iteration
	{
		if (nearOptimal == 0)
			nearOptimal = 1;
		else
			nearOptimal++;
	}
	else
	{
		if (nearOptimal < 3)
			nearOptimal = 0;
	}
	for (int jj = 0; jj < prob.nbFirstVars; ++jj)
		xiterateXf(jj) = stab_center[jj];
	stat.solvetime = clock.getTime()-starttime;
	stab_center.end();
	xvals.end();
	env2.end();
	cplex.end();
	model.end();
	cuts.end();
	center_cons.end();
	x.end();
	stat.finalpartitionsize = partition.size();
	if (option == 1)
		cout << "end solution mode...up to sample error" << endl;
	if (option == 2)
		cout << "end evaluation mode..." << endl;
	return returnflag;
}

void setup_bundle_QP(IloEnv& env, const TSLP& prob, IloCplex& cplex, IloModel& model, IloNumVarArray& x, const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons)
{
	model = IloModel(env);
	// first stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			// Perform refinement
			lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		model.add(range);
		lhs.end();
	}
	// Adding objective
	QPobj = IloMinimize(env);
	model.add(QPobj);
	IloExpr objExpr(env);
	IloNumVarArray y(env,prob.nbFirstVars, 0, IloInfinity);
	for (int j = 0; j < prob.nbFirstVars; ++j)
		objExpr += y[j];
	QPobj.setExpr(objExpr);
	objExpr.end();

	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		IloRange range(env, -IloInfinity, y[j]-x[j], IloInfinity);
		model.add(range);
		center_cons.add(range);
	}

	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		IloRange range(env, -IloInfinity, y[j]+x[j], IloInfinity);
		model.add(range);
		center_cons.add(range);
	}

	// Add cuts
	for (int l = 0; l < cuts.getSize(); ++l)
		model.add(cuts[l]);
	cplex = IloCplex(model);
	cplex.setParam(IloCplex::TiLim,3600);
	cplex.setParam(IloCplex::Threads, 1);
	cplex.setParam(IloCplex::BarDisplay, 0);
	cplex.setParam(IloCplex::SimDisplay, 0);
	cplex.setOut(env.getNullStream());
	//cplex.setParam(IloCplex::EpOpt, 1e-4);
}

double coarse_oracle(IloEnv& env, TSLP& prob, Subprob& subp, vector<Component>& partition, IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, IloCplex& cplex, IloModel& model, const IloNumVarArray& x, STAT& stat, IloRangeArray& center_cons, const IloNumArray& stab_center, IloRangeArray& cuts, const vector<double>& cutrhs, VectorXf& aggrCoarseCut, double& coarseCutRhs, vector<VectorXf>& partcoef, vector<double>& partrhs, double starttime, IloTimer& timer, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option)
{
	// coarse oracle
	// won't add any cuts in this subroutine, just collect information, and decide whether or not add the coarse cut depending on whether the descent target is achieved
	bool cutflag = 1;
	IloNumArray secvarlb(env, partition.size()*prob.nbSecVars);
	IloNumArray secvarub(env, partition.size()*prob.nbSecVars);
	IloNumArray secconstrbd(env, partition.size()*prob.nbSecRows);
	setAggregatedBounds(prob, partition, secvarlb, secvarub, secconstrbd);
	double inner_up = stat.feasobjval;
	double inner_low = stat.relaxobjval;
	double levelobj = prob.kappa*inner_low + (1-prob.kappa)*inner_up;
	// Update all the oracles in the bundle for the QP model
	for (int l = 0; l < cuts.getSize(); ++l)
		cuts[l].setUB(levelobj-cutrhs[l]);
	// Update stablization center
	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		if (fabs(stab_center[j]) > 1e-7)
		{
			center_cons[j].setLB(-stab_center[j]);
			center_cons[j+prob.nbFirstVars].setLB(stab_center[j]);
		}
		else
		{
			center_cons[j].setLB(0);
			center_cons[j+prob.nbFirstVars].setLB(0);
		}
	}
	double totalobjval;
	while (cutflag == 1)
	{
		// Keep going if no opt cut has been added
		cutflag = 0;
		// First of all, solve the master problem
		cplex.solve();
		//cout << "master QP status = " << cplex.getCplexStatus() << endl;
		if (cplex.getStatus() == IloAlgorithm::Infeasible)
		{
			// Infeasible: update inner_low and level obj
			inner_low = levelobj;
			levelobj = prob.kappa*inner_low + (1-prob.kappa)*inner_up;
			for (int l = 0; l < cuts.getSize(); ++l)
				cuts[l].setUB(levelobj-cutrhs[l]);
			stat.relaxobjval = levelobj;
			if (timer.getTime()-starttime > TIMELIMIT)
				break;
			if (fabs(stat.feasobjval-stat.relaxobjval)*1.0/(fabs(stat.feasobjval)+1e-10) > 1e-6)
				cutflag = 1;
			else
			{
				// already optimal
				cutflag = 0;
			}
			continue;
		}
		aggrCoarseCut.setZero();
		cplex.getValues(xvals, x);
		VectorXf xiterateXf(prob.nbFirstVars);
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterateXf(j) = xvals[j];
		// Start generating partition-based Benders cuts
		feasboundscen = 0;
		cutcoefscen.setZero();
		totalobjval = 0;
		for (int i = 0; i < partition.size(); ++i)
		{
			IloNumArray duals(env);
			bool scenfeasflag;
			double subobjval = subprob_partition(subp, secvarlb, secvarub, prob, xvals, duals, partition, i, scenfeasflag);
			VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
			for (int j = 0; j < prob.nbSecRows+prob.nbSecVars; ++j)
				dualvec(j) = duals[j];
			if (scenfeasflag == 1)
			{
				VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
				for (int j = 0; j < prob.nbFirstVars; ++j)
				{
					if (fabs(opt_cut_coef[j]) < 1e-7)
						opt_cut_coef[j] = 0;
				}
				aggrCoarseCut += opt_cut_coef*(partition[i].indices.size());
				totalobjval += subobjval;
				partcoef[i] = opt_cut_coef;
				partrhs[i] = subobjval;
				if (partition[i].indices.size() == 1)
				{
					feasboundscen += subobjval;
					cutcoefscen += opt_cut_coef;
					scenObjs.push_back(subobjval);
					if (option == 1)
					{
						if (addToCollection(dualvec, dualInfoCollection) == true)
						{
							DualInfo dual;
							dual.dualvec = dualvec;
							dual.coefvec = opt_cut_coef;
							dual.rhs = subobjval;
							for (int j = 0; j < prob.nbFirstVars; ++j)
								dual.rhs += opt_cut_coef[j]*xvals[j];
							dual.rhs -= rhsvecs[partition[i].indices[0]].dot(dualvec.segment(0, prob.nbSecRows));
							dualInfoCollection.push_back(dual);
						}
					}
				}
			}
			else
			{
				//cout << "feas cuts!" << endl;
				cutflag = 1;
				// Add feasibility cuts
				add_feas_cuts(env, prob, partition, model, x, xvals, subobjval, dualvec, i);
			}
			duals.end();
		}
		if (cutflag == 0)
		{
			// So it is feasible
			// Don't add immediately, save information
			double lhsval = 0;
			for (int j = 0; j < prob.nbFirstVars; ++j)
			{
				if (fabs(aggrCoarseCut[j]) > 1e-7)
					lhsval += xvals[j]*aggrCoarseCut[j];
			}
			coarseCutRhs = (totalobjval + lhsval)*1.0/samples.size();
			totalobjval = totalobjval*1.0/samples.size();
			for (int j = 0; j < prob.nbFirstVars; ++j)
				totalobjval += prob.objcoef[j]*xvals[j];
		}
	}
	secvarlb.end();
	secvarub.end();
	secconstrbd.end();
	return totalobjval;
}

void setAggregatedBounds(const TSLP& prob, const vector<Component>& partition, IloNumArray& secvarlb, IloNumArray& secvarub, IloNumArray& secconstrbd)
{
	for (int i = 0; i < partition.size(); ++i)
	{
		// set aggregated variable bound according to partition
		for (int j = 0; j < prob.nbSecVars; ++j)
		{
			bool lbinfflag = 0;
			bool ubinfflag = 0;
			double templb = 0.0;
			double tempub = 0.0;
			for (int k = 0; k < partition[i].indices.size(); ++k)
			{
				if (prob.secondvarlb[partition[i].indices[k]*prob.nbSecVars+j] == -IloInfinity)
				{
					lbinfflag = 1;
					break;
				}
				else
					templb += prob.secondvarlb[partition[i].indices[k]*prob.nbSecVars+j];
			}
			for (int k = 0; k < partition[i].indices.size(); ++k)
			{
				if (prob.secondvarub[partition[i].indices[k]*prob.nbSecVars+j] == IloInfinity)
				{
					ubinfflag = 1;
					break;
				}
				else
					tempub += prob.secondvarub[partition[i].indices[k]*prob.nbSecVars+j];
			}
			if (lbinfflag == 1)
				secvarlb[i*prob.nbSecVars+j] = -IloInfinity;
			else
				secvarlb[i*prob.nbSecVars+j] = templb;
			if (ubinfflag == 1)
				secvarub[i*prob.nbSecVars+j] = IloInfinity;
			else
				secvarub[i*prob.nbSecVars+j] = tempub;
		}
		// set second stage constraint bounds according to partition
		for (int d = 0; d < prob.nbSecRows; ++d)
		{
			double tempbd = 0.0;
			for (int k = 0; k < partition[i].indices.size(); ++k)
				tempbd += prob.secondconstrbd[partition[i].indices[k]*prob.nbSecRows+d];
			secconstrbd[i*prob.nbSecRows+d] = tempbd;
		}
	}
}

double subprob_partition(Subprob& subp, IloNumArray& secvarlb, IloNumArray& secvarub, const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, const vector<Component>& partition, int k, bool& feasflag)
{
	// Set constraint bounds
	for (int i = 0; i < prob.nbSecRows; ++i)
	{
		double bd = 0;
		for (int l = 0; l < partition[k].indices.size(); ++l)
		{
			double subbd = prob.secondconstrbd[partition[k].indices[l]*prob.nbSecRows+i];
			for (int j = 0; j < prob.nbPerRow[i]; ++j)
			{
				int ind = prob.CoefInd[i][j];
				if (ind < prob.nbFirstVars)
					subbd -= prob.CoefMat[i][j]*xvals[ind];
			}
			bd += subbd;
		}
		if (prob.secondconstrsense[i] == -1)
			subp.suboptcon[i].setLB(bd);
		if (prob.secondconstrsense[i] == 0)
		{
			if (subp.suboptcon[i].getUB() < bd)
			{
				subp.suboptcon[i].setUB(bd);
				subp.suboptcon[i].setLB(bd);
			}
			else
			{
				subp.suboptcon[i].setLB(bd);
				subp.suboptcon[i].setUB(bd);
			}
		}
		if (prob.secondconstrsense[i] == 1)
			subp.suboptcon[i].setUB(bd);
	}
	// Set variable bounds
	for (int j = 0; j < prob.nbSecVars; ++j)
	{
		if (secvarlb[k*prob.nbSecVars+j] != -IloInfinity)
			subp.suboptcon[prob.nbSecRows+j].setLB(secvarlb[k*prob.nbSecVars+j]);
		else
			subp.suboptcon[prob.nbSecRows+j].setLB(-IloInfinity);
		if (secvarub[k*prob.nbSecVars+j] != IloInfinity)
			subp.suboptcon[prob.nbSecRows+prob.nbSecVars+j].setLB(-secvarub[k*prob.nbSecVars+j]);
		else
			subp.suboptcon[prob.nbSecRows+prob.nbSecVars+j].setLB(-IloInfinity);
	}
	subp.suboptcplex.solve();
	double returnval;
	if (subp.suboptcplex.getStatus() == IloAlgorithm::Optimal)
	{
		returnval = subp.suboptcplex.getObjValue();
		subp.suboptcplex.getDuals(duals, subp.suboptcon);
		feasflag = 1;
	}
	else
	{
		feasflag = 0;
		// infeasible! Get extreme rays
		for (int i = 0; i < prob.nbSecRows; ++i)
		{
			double bd = 0;
			for (int l = 0; l < partition[k].indices.size(); ++l)
			{
				double subbd = prob.secondconstrbd[partition[k].indices[l]*prob.nbSecRows+i];
				for (int j = 0; j < prob.nbPerRow[i]; ++j)
				{
					int ind = prob.CoefInd[i][j];
					if (ind < prob.nbFirstVars)
						subbd -= prob.CoefMat[i][j]*xvals[ind];
				}
				bd += subbd;
			}
			if (prob.secondconstrsense[i] == -1)
				subp.subfeascon[i].setLB(bd);
			if (prob.secondconstrsense[i] == 0)
			{
				if (subp.subfeascon[i].getUB() < bd)
				{
					subp.subfeascon[i].setUB(bd);
					subp.subfeascon[i].setLB(bd);
				}
				else
				{
					subp.subfeascon[i].setLB(bd);
					subp.subfeascon[i].setUB(bd);
				}
			}
			if (prob.secondconstrsense[i] == 1)
				subp.subfeascon[i].setUB(bd);
		}
		for (int j = 0; j < prob.nbSecVars; ++j)
		{
			if (secvarlb[k*prob.nbSecVars+j] != -IloInfinity)
				subp.subfeascon[prob.nbSecRows+j].setLB(secvarlb[k*prob.nbSecVars+j]);
			else
				subp.subfeascon[prob.nbSecRows+j].setLB(-IloInfinity);
			if (secvarub[k*prob.nbSecVars+j] != IloInfinity)
				subp.subfeascon[prob.nbSecRows+prob.nbSecVars+j].setLB(-secvarub[k*prob.nbSecVars+j]);
			else
				subp.subfeascon[prob.nbSecRows+prob.nbSecVars+j].setLB(-IloInfinity);
		}
		subp.subfeascplex.solve();
		subp.subfeascplex.getDuals(duals, subp.subfeascon);
		returnval = subp.subfeascplex.getObjValue();
	}
	return returnval;
}

bool solve_scen_subprobs(IloEnv& env, IloEnv& env2, const TSLP& prob, Subprob& subp, const vector<Component>& partition, const IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, IloModel& model, const IloNumVarArray& x, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<double>& scenObjs, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option)
{
	bool returnflag = 1;
 	for (int i = 0; i < partition.size(); ++i)
	{
		double sum_of_infeas = 0;
		vector<IloNumArray> extreme_points, extreme_rays;
		vector<int> extreme_points_ind, extreme_rays_ind;
		vector< vector<int> > extreme_ray_map; // record groups of extreme rays that are stored in the list extreme_rays, extreme_rays_ind
		// solve subproblems for each partition
		if (partition[i].indices.size() > 1)
		{
			for (int k = 0; k < partition[i].indices.size(); ++k)
			{
				IloNumArray duals(env2);
				bool feasflag;
				double subobjval = subprob(subp, prob, xvals, duals, partition[i].indices[k], feasflag);
				if (feasflag == 1)
				{
					// optimal, so return extreme point solution
					extreme_points.push_back(duals);
					extreme_points_ind.push_back(partition[i].indices[k]);
					feasboundscen += subobjval;
					VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
					for (int j = 0; j < prob.nbSecRows+prob.nbSecVars; ++j)
						dualvec(j) = duals[j];
					VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
					cutcoefscen += opt_cut_coef;
					scenObjs.push_back(subobjval);
					if (option == 1)
					{
						if (addToCollection(dualvec, dualInfoCollection) == true)
						{
							DualInfo dual;
							dual.dualvec = dualvec;
							dual.coefvec = opt_cut_coef;
							dual.rhs = subobjval;
							for (int j = 0; j < prob.nbFirstVars; ++j)
								dual.rhs += opt_cut_coef[j]*xvals[j];
							dual.rhs -= rhsvecs[partition[i].indices[k]].dot(dualvec.segment(0, prob.nbSecRows));
							dualInfoCollection.push_back(dual);
						}
					}
				}
				else
				{
					returnflag = 0;
					//cout << "infeasible scenario subproblem!" << endl;
					extreme_rays.push_back(duals);
					extreme_rays_ind.push_back(partition[i].indices[k]);
					sum_of_infeas += subobjval;
				}
			}
			// Perform refinement
			double refinestart = clock.getTime();
			simple_refine(partition[i], prob, extreme_points, extreme_points_ind, extreme_rays, extreme_rays_ind, new_partition, extreme_ray_map);
			stat.refinetime += clock.getTime()-refinestart;
		}
		else
		{
			// Don't need refine
			new_partition.push_back(partition[i]);
		}
		// Now add feasibility cuts
		for (int j = 0; j < extreme_ray_map.size(); ++j)
		{
			// add feasibility cuts group by group
			gen_feasibility_cuts(env, prob, xvals, extreme_ray_map[j], extreme_rays, extreme_rays_ind, sum_of_infeas, model, x);
			stat.num_feas_cuts++;
		}
		for (int j = 0; j < extreme_points.size(); ++j)
			extreme_points[j].end();
		for (int j = 0; j < extreme_rays.size(); ++j)
			extreme_rays[j].end();
	}
	return returnflag;
}


bool solve_scen_subprobs_target(IloEnv& env, IloEnv& env2, const TSLP& prob, Subprob& subp, const vector<Component>& partition, const IloNumArray& xvals, IloModel& model, const IloNumVarArray& x, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<VectorXf>& partcoef, vector<double>& partrhs, double descent_target, bool& fullupdateflag, double& coarseLB, VectorXf& aggrCoarseCut, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option)
{
	// Solve scenario subproblems component by component, stop when hopeless to achieve descent target
	bool returnflag = 1;
	// Currently: go through all the partition component by exploring the bigger component first
	vector<IndexVal> sizelist;
	for (int ii = 0; ii < partition.size(); ++ii)
	{
		IndexVal indexval;
		indexval.ind = ii;
		indexval.val = partition[ii].indices.size();
		sizelist.push_back(indexval);
	}
	sort(sizelist.begin(), sizelist.end(), less<IndexVal>());
	int ind = 0;
 	while (ind < partition.size())
	{
		int i = sizelist[ind].ind;
		if (partition[i].indices.size() > 1)
		{
			// solve subproblems for each partition
			double sum_of_infeas = 0;
			vector<IloNumArray> extreme_points, extreme_rays;
			vector<int> extreme_points_ind, extreme_rays_ind;
			vector< vector<int> > extreme_ray_map; // record groups of extreme rays that are stored in the list extreme_rays, extreme_rays_ind
			coarseLB -= partrhs[i]*1.0/samples.size();
			aggrCoarseCut -= partcoef[i]*(partition[i].indices.size());
			for (int k = 0; k < partition[i].indices.size(); ++k)
			{
				IloNumArray duals(env2);
				bool feasflag;
				double subobjval = subprob(subp, prob, xvals, duals, partition[i].indices[k], feasflag);
				if (feasflag == 1)
				{
					// optimal, so return extreme point solution
					extreme_points.push_back(duals);
					extreme_points_ind.push_back(partition[i].indices[k]);
					VectorXf dualvec(prob.nbSecRows+prob.nbSecVars);
					for (int j = 0; j < prob.nbSecRows+prob.nbSecVars; ++j)
						dualvec(j) = duals[j];
					VectorXf opt_cut_coef = prob.CoefMatXf.transpose()*dualvec.segment(0,prob.nbSecRows);
					coarseLB += subobjval*1.0/samples.size();
					aggrCoarseCut += opt_cut_coef;
					scenObjs.push_back(subobjval);
					if (option == 1)
					{
						if (addToCollection(dualvec, dualInfoCollection) == true)
						{
							DualInfo dual;
							dual.dualvec = dualvec;
							dual.coefvec = opt_cut_coef;
							dual.rhs = subobjval;
							for (int j = 0; j < prob.nbFirstVars; ++j)
								dual.rhs += opt_cut_coef[j]*xvals[j];
							dual.rhs -= rhsvecs[partition[i].indices[k]].dot(dualvec.segment(0, prob.nbSecRows));
							dualInfoCollection.push_back(dual);
						}
					}
				}
				else
				{
					returnflag = 0;
					cout << "infeasible scenario subproblem!" << endl;
					extreme_rays.push_back(duals);
					extreme_rays_ind.push_back(partition[i].indices[k]);
					sum_of_infeas += subobjval;
				}
			}
			// Perform refinement
			double refinestart = clock.getTime();
			simple_refine(partition[i], prob, extreme_points, extreme_points_ind, extreme_rays, extreme_rays_ind, new_partition, extreme_ray_map);
			stat.refinetime += clock.getTime()-refinestart;
			// Now add feasibility cuts
			for (int j = 0; j < extreme_ray_map.size(); ++j)
			{
				// add feasibility cuts group by group
				gen_feasibility_cuts(env, prob, xvals, extreme_ray_map[j], extreme_rays, extreme_rays_ind, sum_of_infeas, model, x);
				stat.num_feas_cuts++;
			}
			for (int j = 0; j < extreme_points.size(); ++j)
				extreme_points[j].end();
			for (int j = 0; j < extreme_rays.size(); ++j)
				extreme_rays[j].end();
			if (coarseLB > descent_target)
			{
				// If hopeless to achieve the descent target, break out of loop
				for (int j = ind+1; j < partition.size(); ++j)
					new_partition.push_back(partition[sizelist[j].ind]);
				break;
			}
		}
		else
		{
			// Don't need refine
			new_partition.push_back(partition[i]);
		}
		ind++;
	}
	if (ind >= partition.size()-1 || partition[sizelist[ind+1].ind].indices.size() == 1)
	{
		// All scenario subproblems have been explored: either finish, or break at the last iteration, or after break, all partition components have size = 1
		fullupdateflag = 1;
	}
	return returnflag;
}

void add_feas_cuts(IloEnv& env, TSLP& prob, const vector<Component>& partition, IloModel& model, const IloNumVarArray& x, const IloNumArray& xvals, double subobjval, const VectorXf& dualvec, int i)
{
	// Add feasibility cuts
	vector<double> feas_cut_coef(prob.nbFirstVars, 0);
	double sum_xvals = 0.0;
	for (int ii = 0; ii < prob.nbSecRows; ++ii)
	{
		for (int j = 0; j < prob.nbPerRow[ii]; ++j)
		{
			if (prob.CoefInd[ii][j] < prob.nbFirstVars)
			{
				feas_cut_coef[prob.CoefInd[ii][j]] += prob.CoefMat[ii][j]*dualvec[ii]*partition[i].indices.size();
				sum_xvals += prob.CoefMat[ii][j]*dualvec[ii]*partition[i].indices.size()*xvals[prob.CoefInd[ii][j]];
			}
		}
	}
	IloExpr lhsfeas(env);
	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		if (fabs(feas_cut_coef[j]) > 1e-7)
			lhsfeas += feas_cut_coef[j]*x[j];
	}
	model.add(lhsfeas >= subobjval+sum_xvals);
	lhsfeas.end();
}

void solve_adaptive_partition(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, int option)
{
	// option = 0: B&P-L with 1e-6 precision for solving SAA problems
	// option = 1,2: B&P-L FSP/SSP
	// option = 3: use fixed rate schedule on increasing sample size, when BM fails to progress
	// option = 4: use fixed rate schedule all the way
	// option = 5: use a simple heuristic "trust region" idea to adjust the sample size increasing rate
	double alpha = 0.1;
	Sequence seq;
	sequentialSetup(seq, option);
	double epsilon; // epsilon is for stopping criterion of B&P-L
	bool iterflag = 1;
	int iter = 0;

	// Construct subproblems
	Subprob subp;
	construct_second_opt(env, prob, subp);
	construct_second_feas(env, prob, subp);

	// collection of dual multipliers
	vector<DualInfo> dualInfoCollection;
	VectorXf xiterateXf(prob.nbFirstVars);
	vector<VectorXf> rhsvecs;
	// Store all the rhs vectors
	for (int k = 0; k < prob.nbScens; ++k)
	{
		VectorXf rhsXf(prob.nbSecRows);
		for (int j = 0; j < prob.nbSecRows; ++j)
			rhsXf[j] = prob.secondconstrbd[j+k*prob.nbSecRows];
		rhsvecs.push_back(rhsXf);
	}
	int nearOptimal = 0; // keep track of # of consecutive times where the new sampled problem is nearly optimal at the very first iteration

	int nbIterEvalScens, nbIterSolScens;
	double starttime = clock.getTime();

	// We create some safeguard in case we do not finish by satisfying the stopping criterion in nMax steps
	double minGS = 1e8;
	double minGCI = 1e8;
	double G, S;
	//double increaseRate = 0.1;
	while (iterflag == 1) // Outer loop
	{
		/* Setting up the sample size */
		cout << "outer iter = " << iter << ", dualInfoCollection.size = " << dualInfoCollection.size() << ", increaseRate = " << prob.increaseRate << endl;
		if (iter == 0 || option == 0 || option == 1 || option == 2)
			nbIterEvalScens = seq.sampleSizes[iter];
		else
		{
			if (option == 4 || option == 5)
			{
				// option = 4: use a fixed rate all the way
				// option = 5: use a simple heuristic trust region idea to choose the sample size increasing rate
				nbIterEvalScens = int(nbIterEvalScens * (1+prob.increaseRate));
				if (nbIterEvalScens < seq.sampleSizes[iter])
					nbIterEvalScens = seq.sampleSizes[iter];
			}
			// use fixed rate schedule on increasing sample size, when BM fails to progress
			if (option == 3)
			{
				if (nearOptimal >= 3)
				{
					nbIterEvalScens = int(nbIterEvalScens * (1+prob.increaseRate));
					if (nbIterEvalScens < seq.sampleSizes[iter])
						nbIterEvalScens = seq.sampleSizes[iter];
				}
				else
				{
					if (seq.sampleSizes[iter] > nbIterEvalScens)
						nbIterEvalScens = seq.sampleSizes[iter];
				}
			}
		}
		nbIterSolScens = 2*nbIterEvalScens;
		cout << "nbIterSolScens = " << nbIterSolScens << endl;

		/* Begin Solving sampled problems */

		vector<int> samplesForSol(nbIterSolScens);
		for (int j = 0; j < nbIterSolScens; ++j)
			samplesForSol[j] = rand() % prob.nbScens;
		STAT tempstat0;
		tempstat0.relaxobjval = -1e10;
		tempstat0.feasobjval = 1e10;
		tempstat0.solvetime = 0;
		tempstat0.warmstarttime = 0;
		tempstat0.warmstartcuttime = 0;
		tempstat0.iter = 0;

		bool solFlag;

		stat.solvetime = clock.getTime()-starttime;
		double remaintime = TIMELIMIT-stat.solvetime;
		if (remaintime < 1)
		{
			cout << "TIME ERROR!" << endl;
			exit(0);
		}
		if (option != 0)
		{
			if (iter == 0)
			{
				// stop when opt gap is small relative to the sample error, the very first sampled problem (no stability center provided)
				solFlag = solve_partly_inexact_bundle(env, prob, subp, tempstat0, clock, samplesForSol, xiterateXf, 1, 1, dualInfoCollection, rhsvecs, nearOptimal, remaintime);
			}
			else
			{
				// stop when opt gap is small relative to the sample error, there exists a current stability center
				solFlag = solve_partly_inexact_bundle(env, prob, subp, tempstat0, clock, samplesForSol, xiterateXf, 1, 0, dualInfoCollection, rhsvecs, nearOptimal, remaintime);
			}
		}
		else
		{
			// solve to 1e-6
			if (iter == 0)
			{
				// stop when opt gap is small relative to the sample error, the very first sampled problem (no stability center provided)
				solFlag = solve_partly_inexact_bundle(env, prob, subp, tempstat0, clock, samplesForSol, xiterateXf, 0, 1, dualInfoCollection, rhsvecs, nearOptimal, remaintime);
			}
			else
			{
				// stop when opt gap is small relative to the sample error, there exists a current stability center
				solFlag = solve_partly_inexact_bundle(env, prob, subp, tempstat0, clock, samplesForSol, xiterateXf, 0, 0, dualInfoCollection, rhsvecs, nearOptimal, remaintime);
			}
		}

		if (solFlag == 0)
		{
			// problem did not get solved most likely because of time limit, abort using previous stat
			iterflag = 0;
		}

		else
		{
			cout << "# of inner iterations = " << tempstat0.iter << endl;
			if (iter > 0 && option == 5)
			{
				// option = 5: use a simple heuristic "trust region" idea to adjust the sample size increasing rate
				if (tempstat0.iter <= 2)
				{
					// increase the increasing rate
					prob.increaseRate = 2*prob.increaseRate;
				}
				else
				{
					if (tempstat0.iter > 4)
					{
						// decrease the increasing rate
						prob.increaseRate = prob.increaseRate*0.5;
					}
					else
					{
						// otherwise, i.e., inner iter = 3, 4, stay the same
					}
				}
				if (prob.increaseRate < 0.05)
					prob.increaseRate = 0.05;
				if (prob.increaseRate > 2)
					prob.increaseRate = 2;
			}
			stat.iter += tempstat0.iter; // stat.iter records the total number of inner iterations
			stat.warmstarttime += tempstat0.warmstarttime;
			stat.warmstartcuttime += tempstat0.warmstartcuttime;
			stat.mastertime += tempstat0.solvetime; // stat.mastertime records the solution time

			/* Begin Evaluation */
			IloNumArray xvals(env, prob.nbFirstVars);
			for (int j = 0; j < prob.nbFirstVars; ++j)
				xvals[j] = xiterateXf(j);
			// Now obtained xiterateXf, xvals, scenobj corresponding to \hat{x}_k
			if (iter == 0)
			{
				// epsilon is for stopping criterion of B&P-L, just set it to be small enough relative to the initial UB obtained from the first iteration
				epsilon = prob.eps*fabs(tempstat0.feasobjval);
			}
			// Test stopping criterion: SRP type CI estimation
			double tempEvaltime = clock.getTime();
			G = 0;
			S = 0;
			vector<double> scenObjEval(nbIterEvalScens);
			vector<int> samplesForEval(nbIterEvalScens);
			for (int j = 0; j < nbIterEvalScens; ++j)
				samplesForEval[j] = rand() % prob.nbScens;
			STAT tempstat;
			tempstat.iter = 0;
			tempstat.relaxobjval = -1e10;
			tempstat.feasobjval = 1e10;
			VectorXf xiterateXf2 = xiterateXf;
			int nearOptimal2 = 0;

			stat.solvetime = clock.getTime()-starttime;
			remaintime = TIMELIMIT-stat.solvetime;

			// Use the relative opt gap, evaluation mode: 1e-4 as the threshold, there exists a current stabilization center
			solFlag = solve_partly_inexact_bundle(env, prob, subp, tempstat, clock, samplesForEval, xiterateXf2, 2, 0, dualInfoCollection, rhsvecs, nearOptimal2, remaintime);
			if (solFlag == 0)
			{
				// problem did not get solved most likely because of time limit, abort using previous stat
				iterflag = 0;
			}
			else
			{
				stat.finalSampleSize = nbIterSolScens;
				IloNumArray xvals2(env, prob.nbFirstVars);
				for (int j = 0; j < prob.nbFirstVars; ++j)
					xvals2[j] = xiterateXf2(j);
				// xiterateXf, xvals, scenObj correspondiing to \hat{x}_k, xiterateXf2, xvals2, scenObj2 corresponding to x^*_{n_k}
				SRP(env, prob, subp, clock, nbIterEvalScens, samplesForEval, G, S, xvals, xvals2, scenObjEval);
				xvals2.end();
				stat.evaltime += (clock.getTime()-tempEvaltime);

				// Check if the stopping criterion is met
				if (option == 0 || option == 1 || option == 2)
				{
					cout << "G = " << G << ", S = " << S << ", CI width = " << G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) << ", epsilon = " << epsilon << endl;
					// B&P-L: FSP/SSP
					if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) < minGCI)
						minGCI = G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens);
					if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) <= epsilon)
					{
						iterflag = 0;
						stat.finalSampleSize = nbIterSolScens;
					}
					else
					{
						iter++;
						if (option == 2)
						{
							// sample size for the next iteration will depend (adaptively) on the statistics of the current iteration
							double const_b = t_quant*S+1;
							double const_c = nbIterEvalScens*G;
							double const_delta = const_b*const_b + 4*epsilon*const_c;
							double const_v = (const_b + sqrt(const_delta))*1.0/(2*epsilon);
							seq.sampleSizes[iter] = int(const_v*const_v)+1;
						}
					}
				}
				if (option == 3 || option == 4 || option == 5)
				{
					// Start with BM - if BM fails to progress - switch to fixed (exponential) rate schedule
					// Use either B&M (2011) or B&P-L criteria
					/*
					if (G*1.0/S < minGS)
						minGS = G*1.0/S;
					if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) < minGCI)
						minGCI = G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens);
					if (G <= seq.h2*S + seq.eps2)
					{
						cout << "G/S = " << G*1.0/S << ", G = " << G << ", S = " << S << endl;
						iterflag = 0;
					}
					*/
					cout << "G = " << G << ", S = " << S << ", CI width = " << G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) << ", epsilon = " << epsilon << endl;
					// B&P-L
					if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) < minGCI)
						minGCI = G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens);
					if (G + (t_quant*S+1)*1.0/sqrt(nbIterEvalScens) <= epsilon)
					{
						iterflag = 0;
						stat.finalSampleSize = nbIterSolScens;
					}
					else
						iter++;
				}
				stat.solvetime = clock.getTime()-starttime;
				if (iter == seq.nMax || stat.solvetime > TIMELIMIT)
				{
					iterflag = 0;
					stat.finalSampleSize = nbIterSolScens;
				}
			}
			xvals.end();
		}
	}

	// Now check if this estimation is correct by evaluating \hat{x}_k using the true distribution (all samples)
	finalEval(env, prob, subp, xiterateXf, stat);
	stat.mainIter = iter;
	// B&P-L: FSP or SSP
	stat.gapThreshold = minGCI;
	cout << "optimality gap is less than " << stat.gapThreshold << " with prob >= 90%" << endl;
	subp.suboptcplex.end();
	subp.suboptmodel.end();
	subp.suboptcon.end();
	subp.subopty.end();
	subp.subfeascplex.end();
	subp.subfeasmodel.end();
	subp.subfeascon.end();
	subp.subfeasy.end();
}

void computeSamplingError(double& samplingError, const vector<double>& scenObjs)
{
	samplingError = 0;
	double sampleMean = 0;
	for (int k = 0; k < scenObjs.size(); ++k)
		sampleMean += scenObjs[k];
	sampleMean = sampleMean*1.0/scenObjs.size();
	for (int k = 0; k < scenObjs.size(); ++k)
		samplingError += pow(scenObjs[k]-sampleMean, 2);
	samplingError = sqrt(samplingError)*1.0/scenObjs.size();
	// \delta = 1e-3
	if (samplingError < 1e-3*1.0/sqrt(scenObjs.size()))
		samplingError = 1e-3*1.0/sqrt(scenObjs.size());
}

double solve_warmstart(IloEnv& env, const TSLP& prob, const vector<int>& samples, const IloNumArray& stab_center, const vector<DualInfo>& dualInfoCollection, vector< vector<double> >& cutcoefs, vector<double>& cutrhs, vector<Component>& partition, const vector<VectorXf>& rhsvecs, IloNumArray& xvals, IloTimer & clock, STAT& stat)
{
	IloModel model(env);
	IloNumVarArray x(env, prob.firstvarlb, prob.firstvarub);
	IloNumVar theta(env);
	// first stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			// Perform refinement
			lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		model.add(range);
		lhs.end();
	}
	IloExpr obj(env);
	for (int i = 0; i < prob.nbFirstVars; ++i)
		obj += x[i]*prob.objcoef[i];
	obj += theta*1.0/(samples.size());
	model.add(IloMinimize(env, obj));
	obj.end();
	IloCplex cplex(model);
	cplex.setParam(IloCplex::TiLim,3600);
	cplex.setParam(IloCplex::Threads, 1);
	cplex.setParam(IloCplex::BarDisplay, 0);
	cplex.setParam(IloCplex::SimDisplay, 0);
	cplex.setOut(env.getNullStream());
	// Assembly the matrix to do matrix-vector multiplication instead of componentwise inner products
	int nbRows = dualInfoCollection.size();
	MatrixXf dualvecMatrix(nbRows,prob.nbSecRows);
	MatrixXf coefvecMatrix(nbRows,prob.nbFirstVars);
	VectorXf rhsVector(nbRows);
	for (int l = 0; l < dualInfoCollection.size(); ++l)
	{
		dualvecMatrix.row(l) = dualInfoCollection[l].dualvec;
		coefvecMatrix.row(l) = dualInfoCollection[l].coefvec;
		rhsVector(l) = dualInfoCollection[l].rhs;
	}
	// Initial cut
	VectorXf xiterate(prob.nbFirstVars);
	for (int j = 0; j < prob.nbFirstVars; ++j)
		xiterate[j] = stab_center[j];
	VectorXf initialCutCoef(prob.nbFirstVars);
	initialCutCoef.setZero();
	double initialCutRhs = 0;
	double tempTime = clock.getTime();
	for (int k = 0; k < samples.size(); ++k)
	{
		// Go through all scenarios
		VectorXf vals = dualvecMatrix*rhsvecs[samples[k]]-coefvecMatrix*xiterate+rhsVector;
		int maxind;
		double maxval = vals.maxCoeff(&maxind);
		/*
		double maxval = -1e8;
		int maxind = -1;
		for (int l = 0; l < dualInfoCollection.size(); ++l)
		{
			double tempval = dualInfoCollection[l].dualvec.dot(rhsvecs[samples[k]]);
			tempval -= dualInfoCollection[l].coefvec.dot(xiterate);
			tempval += dualInfoCollection[l].rhs;
			if (tempval > maxval)
			{
				maxval = tempval;
				maxind = l;
			}
		}
		if (maxind == -1)
		{
			cout << "something is wrong in warmstart!" << endl;
			exit(0);
		}
		*/
		initialCutCoef = initialCutCoef + dualInfoCollection[maxind].coefvec;
		initialCutRhs += dualInfoCollection[maxind].dualvec.dot(rhsvecs[samples[k]])+dualInfoCollection[maxind].rhs;
	}
	stat.warmstartcuttime += clock.getTime()-tempTime;
	IloExpr initialCut(env);
	vector<double> tempcutcoef(prob.nbFirstVars,0);
	for (int i = 0; i < prob.nbFirstVars; ++i)
	{
		if (fabs(initialCutCoef[i]) > 1e-7)
		{
			initialCut += x[i]*initialCutCoef[i];
			tempcutcoef[i] = initialCutCoef[i];
		}
	}
	cutcoefs.push_back(tempcutcoef);
	cutrhs.push_back(initialCutRhs);
	initialCut += theta;
	model.add(initialCut >= initialCutRhs);
	initialCut.end();
	// Loop
	bool loopflag = 1;
	double returnval;
	int nCuts = 0;
	while (loopflag)
	{
		loopflag = 0;
		cplex.solve();
		returnval = cplex.getObjValue();
		cplex.getValues(xvals, x);
		for (int j = 0; j < prob.nbFirstVars; ++j)
			xiterate[j] = xvals[j];
		double thetaval = cplex.getValue(theta);
		VectorXf CutCoef(prob.nbFirstVars);
		CutCoef.setZero();
		double CutRhs = 0;
		tempTime = clock.getTime();
		for (int k = 0; k < samples.size(); ++k)
		{
			// Go through all scenarios
			/*
			double maxval = -1e8;
			int maxind = -1;
			for (int l = 0; l < dualInfoCollection.size(); ++l)
			{
				double tempval = dualInfoCollection[l].dualvec.dot(rhsvecs[samples[k]]);
				tempval -= dualInfoCollection[l].coefvec.dot(xiterate);
				tempval += dualInfoCollection[l].rhs;
				if (tempval > maxval)
				{
					maxval = tempval;
					maxind = l;
				}
			}
			if (maxind == -1)
			{
				cout << "something is wrong in warmstart!" << endl;
				exit(0);
			}
			*/
			VectorXf vals = dualvecMatrix*rhsvecs[samples[k]]-coefvecMatrix*xiterate+rhsVector;
			int maxind;
			double maxval = vals.maxCoeff(&maxind);
			CutCoef = CutCoef + dualInfoCollection[maxind].coefvec;
			CutRhs += dualInfoCollection[maxind].dualvec.dot(rhsvecs[samples[k]])+dualInfoCollection[maxind].rhs;
		}
		stat.warmstartcuttime += clock.getTime()-tempTime;
		double cutlhsval = 0;
		IloExpr Cut(env);
		for (int i = 0; i < prob.nbFirstVars; ++i)
		{
			if (fabs(CutCoef[i]) > 1e-7)
			{
				Cut += x[i]*CutCoef[i];
				cutlhsval += xvals[i]*CutCoef[i];
			}
		}
		Cut += theta;
		cutlhsval += thetaval;
		if (cutlhsval < CutRhs - max(1e-2, abs(CutRhs) * 1e-5))
		{
			model.add(Cut >= CutRhs);
			loopflag = 1;
			vector<double> tempcutcoef(prob.nbFirstVars,0);
			for (int i = 0; i < prob.nbFirstVars; ++i)
			{
				if (fabs(CutCoef[i]) > 1e-7)
					tempcutcoef[i] = CutCoef[i];
			}
			cutcoefs.push_back(tempcutcoef);
			cutrhs.push_back(CutRhs);
			nCuts++;
		}
		Cut.end();
	}
	cout << "# of cuts added in the warmstart = " << nCuts << endl;
	cplex.end();
	model.end();
	x.end();
	theta.end();
	return returnval;
}

void SRP(IloEnv& env, TSLP& prob, Subprob& subp, IloTimer& clock, int nbIterEvalScens, const vector<int>& samplesForEval, double& G, double& S, const IloNumArray& xvals, const IloNumArray& xvals2, vector<double>& scenObjEval)
{
	vector<double> scenObj2(nbIterEvalScens);
	for (int k = 0; k < nbIterEvalScens; ++k)
	{
		// solve subproblems for each scenario
		IloNumArray duals(env);
		bool feasflag;
		double subobjval = subprob(subp, prob, xvals, duals, samplesForEval[k], feasflag);
		if (feasflag == 0)
		{
			cout << "something is wrong!" << endl;
			exit(0);
		}
		IloNumArray duals2(env);
		double subobjval2 = subprob(subp, prob, xvals2, duals2, samplesForEval[k], feasflag);
		if (feasflag == 0)
		{
			cout << "something is wrong!" << endl;
			exit(0);
		}
		duals.end();
		duals2.end();
		scenObjEval[k] = subobjval;
		scenObj2[k] = subobjval2;
	}

	// Use n_k (independent) samples to evaluate this candidate solution: compute G and S
	for (int k = 0; k < nbIterEvalScens; ++k)
		G += (scenObjEval[k]-scenObj2[k]);
	G = G*1.0/nbIterEvalScens;
	double deterGap = 0;
	for (int j = 0; j < prob.nbFirstVars; ++j)
		deterGap += prob.objcoef[j]*(xvals[j]-xvals2[j]);
	G += deterGap;
	for (int k = 0; k < nbIterEvalScens; ++k)
		S += pow(scenObjEval[k]-scenObj2[k]+deterGap-G,2);
	S = S*1.0/(nbIterEvalScens-1);
	S = sqrt(S);
}

void BMschedule(Sequence& seq)
{
	double alpha = 0.1;
	// Now calculate cp
	double cp = 0;
	for (int j = 1; j < 10000; ++j)
		cp += pow(j, -seq.p*log(j));
	cp = cp*1.0/(alpha*sqrt(2*3.14159265));
	cp = 2*log(cp);
	if (cp < 1)
		cp = 1;
	for (int k = 0; k < seq.nMax; ++k)
		seq.sampleSizes[k] = int((cp+2*seq.p*pow(log(k+1),2))*1.0/(pow(seq.h-seq.h2,2)))+1;
}

void sequentialSetup(Sequence& seq, int option)
{
	// The sample size increasing schedule follows option 0: B&M (2011); 1: B&P-L FSP; 2: B&P-L SSP; option >= 3: geometric sequence, but lower bounded by B&M
	// At most 100 outer iterations, nMax and p are pairwise, used in B&M (2011)
	seq.nMax = 100; // int nMax = 500
	seq.p = 0.153; // double p = 0.104
	seq.h = 0.302;
	seq.h2 = 0.015;
	// eps, eps2 are for stopping criteria of B&M
	seq.eps = 2e-7;
	seq.eps2 = 1e-7;
	seq.sampleSizes = vector<int>(seq.nMax);
	//if (option == 0 || option >= 3)
	//	BMschedule(seq);
	seq.sampleSizes[0] = 50;
	int initSample = 50;
	if (option == 1 || option == 0)
	{
		// B&P-L: FSP, linear schedule
		for (int k = 0; k < seq.nMax; ++k)
			//seq.sampleSizes[k] = initSample+10*k;
			seq.sampleSizes[k] = initSample+100*k;
	}
	if (option == 2)
	{
		// B&P-L: SSP
		seq.sampleSizes[0] = initSample;
	}
}

void finalEval(IloEnv& env, TSLP& prob, Subprob& subp, const VectorXf& xiterateXf, STAT& stat)
{
	IloNumArray finalxvals(env);
	double finalObj = 0;
	for (int j = 0; j < prob.nbFirstVars; ++j)
	{
		finalxvals.add(xiterateXf(j));
		finalObj += prob.objcoef[j]*xiterateXf(j);
	}
	double secStageObj = 0;
	for (int k = 0; k < prob.nbScens; ++k)
	{
		IloNumArray duals(env);
		bool feasflag = 0;
		double subobjval = subprob(subp, prob, finalxvals, duals, k, feasflag);
		if (feasflag == 0)
		{
			cout << "something is wrong!" << endl;
			exit(0);
		}
		duals.end();
		secStageObj += subobjval;
	}
	secStageObj = secStageObj*1.0/prob.nbScens;
	finalObj += secStageObj;
	cout << "final Obj = " << finalObj << endl;
	finalxvals.end();
	stat.finalSolExactObj = finalObj;
}
