#pragma once
#ifndef PARTITION_H
#define PARTITION_H

#include "adaptiveSamples.h"
#include "Subproblem.h"
#include "CoarseOracle_Fun.h"
#include "SolveScen_Fun.h"

using namespace std;
// using Eigen::VectorXf;


class Partition {

private:
	vector<Component> partition;
	masterProblem masterProb;
	Subproblem subProb;

// Added helper Functions
bool calculate_refinement(vector<IloNumArray>& extreme_points, vector<IloNumArray>& extreme_rays, vector<int>& extreme_points_ind, vector<int>& extreme_rays_ind, double& feasboundscen, VectorXf& cutcoefscen, vector<double>& scenObjs, int option, vector<DualInfo>& dualInfoCollection, const IloNumArray& xvals, const vector<VectorXf>& rhsvecs, double& sum_of_infeas, /*Subproblem*/);
void add_feasibility_cuts(IloEnv& env, const TSLP& prob, const IloNumArray& xvals, IloModel& model, const IloNumVarArray& x, STAT& stat);


public:
	Partition(masterProblem m, Subprooblem s);

	~Partition();

	double solve_warmstart(IloEnv& env, const TSLP& prob, const vector<int>& samples, const IloNumArray& stab_center, const vector<DualInfo>& dualInfoCollection, vector< vector<double> >& cutcoefs, vector<double>& cutrhs, vector<Component>& partition, const vector<VectorXf>& rhsvecs, IloNumArray& xvals, IloTimer& clock, STAT& stat);

	double coarse_oracle(IloEnv& env, TSLP& prob, Subprob& subp, vector<Component>& partition, IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, IloCplex& cplex, IloModel& model, const IloNumVarArray& x, STAT& stat, IloRangeArray& center_cons, const IloNumArray& stab_center, IloRangeArray& cuts, const vector<double>& cutrhs, VectorXf& aggrCoarseCut, double& coarseCutRhs, vector<VectorXf>& partcoef, vector<double>& partrhs, double starttime, IloTimer& timer, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);

	bool refine_full(IloEnv& env, IloEnv& env2, const TSLP& prob, Subprob& subp, const vector<Component>& partition, const IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, IloModel& model, const IloNumVarArray& x, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<double>& scenObjs, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);
	// formerly solve_scen_subprobs and solve_scen_subprobs_target
	bool refine_part(IloEnv& env, IloEnv& env2, const TSLP& prob, Subprob& subp, const vector<Component>& partition, const IloNumArray& xvals, IloModel& model, const IloNumVarArray& x, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<VectorXf>& partcoef, vector<double>& partrhs, double descent_target, bool& fullupdateflag, double& coarseLB, VectorXf& aggrCoarseCut, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);

	void computeSamplingError(double& samplingError, const vector<double>& scenObjs);

};

#endif
