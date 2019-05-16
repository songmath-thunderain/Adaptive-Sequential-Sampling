/*- mode: C++;
 * Date: Jan. 8, 2019
  // Note: recently changed sequentialSetup function; got rid of evalFlag; added \delta = 1e-3; changed the adaptive rate selection
 */

#include <iostream>
#include <ilcplex/ilocplex.h>
#include "adaptiveSamples.h"
#include <Eigen>
#include <vector>
#include <algorithm>
#include <cmath>
extern "C"{
#include <stdio.h>
#include <stdlib.h>
}
#include <time.h>

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;

double t_quant = 1.282; // For simplicity, just use z quantile (independent of degree of freedom), since sample sizes are usually large in the end

double TIMELIMIT = 7200;

int main(int argc, char **argv)
{
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
	cerr << "usage:  " << argv[0] << " <file>" << endl;
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
		solve_extended(env, prob, stat, clock);
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



