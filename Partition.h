#pragma once
#ifndef PARTITION_H
#define PARTITION_H

#include "Subproblem.h"
#include "Masterproblem.h"

using namespace std;
using Eigen::VectorXf;


class Partition {

private:
	vector<Component> partition;
	//Coarse Oracle's Helper Functions
	void setAggregatedBounds(const TSLP& prob, IloNumArray& secvarlb, IloNumArray& secvarub, IloNumArray& secconstrbd);
	bool addToCollection(const VectorXf& dualvec, vector<DualInfo>& dualInfoCollection);
	void add_feas_cuts(IloEnv& env, TSLP& prob, const IloNumArray& xvals, double subobjval, const VectorXf& dualvec, int i);
	//Solve Scen Subprobs' Helper Functions
	void simple_refine(const Component& component, const TSLP& prob, const vector<IloNumArray>& extreme_points, const vector<int>& extreme_points_ind, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, vector<Component>& new_partition, vector< vector<int> >& extreme_ray_map);
	bool compare_arrays(const TSLP& prob, const IloNumArray& array1, const IloNumArray& array2);

public:
	//Constructor
	Partition();
	//Deconstructor
	~Partition();

	void gen_feasibility_cuts(IloEnv& env, const TSLP& prob, const IloNumArray& xvals, const vector<int>& extreme_ray_map, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, const double sum_of_infeas, Masterproblem& masterProb);
	double solve_warmstart(IloEnv& env, const TSLP& prob, const vector<int>& samples, const IloNumArray& stab_center, const vector<DualInfo>& dualInfoCollection, vector< vector<double> >& cutcoefs, vector<double>& cutrhs, const vector<VectorXf>& rhsvecs, IloNumArray& xvals, Masterproblem& masterProb, IloTimer& clock, STAT& stat);
	double coarse_oracle(IloEnv& env, TSLP& prob, IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, STAT& stat, IloRangeArray& center_cons, const IloNumArray& stab_center, IloRangeArray& cuts, const vector<double>& cutrhs, VectorXf& aggrCoarseCut, double& coarseCutRhs, vector<VectorXf>& partcoef, vector<double>& partrhs, double starttime, IloTimer& timer, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option, Masterproblem& masterProb, Subproblem& subProb);
	bool refine_full(IloEnv& env, IloEnv& env2, const TSLP& prob, const IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<double>& scenObjs, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option, Masterproblem& masterProb, Subproblem& subProb);
	bool refine_part(IloEnv& env, IloEnv& env2, const TSLP& prob, const IloNumArray& xvals, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<VectorXf>& partcoef, vector<double>& partrhs, double descent_target, bool& fullupdateflag, double& coarseLB, VectorXf& aggrCoarseCut, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option, Masterproblem& masterProb, Subproblem& subProb);

	// Allows Solution class to access private data
	friend class Solution;
};

#endif
