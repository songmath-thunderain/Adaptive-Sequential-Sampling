/*- mode: C++;
  Dec. 20, 2017
 */

#include <ilcplex/ilocplex.h>
#include <vector>
#include <set>
#include <Eigen>

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;

typedef IloArray<IloNumArray> NumMatrix;
typedef IloArray<IloIntArray> IntMatrix;
struct TSLP
{
    int nbRows;
	int nbFirstVars;
	int nbSecVars;
    int nbScens;
    IntMatrix firstconstrind;
	NumMatrix firstconstrcoef;
	IntMatrix secondconstrind;
	NumMatrix secondconstrcoef;
	IloNumArray firstconstrlb;
	IloNumArray firstconstrub;
	IloNumArray secondconstrlb;
	IloNumArray secondconstrub;
	IloNumArray secondconstrbd;
	IloNumArray varlb;
	IloNumArray varub;
	IloNumArray firstvarlb;
	IloNumArray firstvarub;
	IloNumArray secondvarlb;
	IloNumArray secondvarub;
	IloNumArray objcoef;
	int nbFirstRows;
	int nbSecRows;
	IloIntArray nbPerRow;
	NumMatrix CoefMat;
	IntMatrix CoefInd;
	int master_solver;
	double distinct_par;
	IntMatrix addfirstconstrind;
	NumMatrix addfirstconstrcoef;
	IloNumArray addfirstconstrrhs;
	IloIntArray secondconstrsense;
	int randomseed;
	MatrixXf CoefMatXf;
	double kappa;
	double kappaf;
	double gamma;
	double increaseRate;
	double eps;
};

struct Sequence
{
	int nMax;
	double p, h, h2, eps, eps2;
	vector<int> sampleSizes;
};

struct Model
{
    IloNumVarArray y;
    IloNumArray lambda;
	IloNumVarArray x;
	IloNumVarArray eta;
};
 
struct STAT
{
	double solvetime;
	double mastertime;
	double qptime;
	double subtime;
	double evaltime;
	double warmstarttime;
	double warmstartcuttime;
	double refinetime;
	double relaxobjval;
	double feasobjval;
	double objval;
	int num_feas_cuts;
	int num_opt_cuts;
	int iter;
	int mainIter;
	double finalSolExactObj;
	double gapThreshold;
	int finalSampleSize;
	int nbSecLPSolves;
	int partitionsize;
	int finalpartitionsize;
};

struct Subprob
{
	IloModel suboptmodel;
	IloRangeArray suboptcon;
	IloNumVarArray subopty;
	IloCplex suboptcplex;
	IloModel subfeasmodel;
	IloRangeArray subfeascon;
	IloNumVarArray subfeasy;
	IloCplex subfeascplex;
};

struct DualInfo
{
	VectorXf dualvec;
	VectorXf coefvec;
	double rhs;
};


struct Component
{
	vector<int> indices;
};

class IndexVal {
public:
	int ind;
	double val;
	IndexVal() {}
	IndexVal(const int& i, const double& v) 
		{ ind = i; val = v; }
};

bool operator<(const IndexVal& a, const IndexVal& b) {
	if (a.val > b.val)
		return true;
	else if (a.val < b.val)
		return false;
	else if (a.ind > b.ind)
		return true;
	else
		return false;
}

bool operator>(const IndexVal& a, const IndexVal& b) {
	if (a.val < b.val)
		return true;
	else if (a.val > b.val)
		return false;
	else if (a.ind < b.ind)
		return true;
	else
		return false;
}

double subprob(Subprob& subp, const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, int k, bool& feasflag);

void gen_feasibility_cuts(IloEnv& env, TSLP& prob, const IloNumArray& xvals, const vector<int>& extreme_ray_map, const vector<IloNumArray>& extreme_rays, const vector<int>& extreme_rays_ind, const double sum_of_infeas);

void preprocessing(IloEnv& env, TSLP& prob);

void construct_second_opt(IloEnv& env, const TSLP& prob, Subprob& subprob);

void construct_second_feas(IloEnv& env, const TSLP& prob, Subprob& subprob);

bool compare_arrays_weighted(const TSLP& prob, const IloNumArray& array1, int a, const IloNumArray& array2, int b);

bool compare_arrays(const TSLP& prob, const IloNumArray& array1, const IloNumArray& array2);

double solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples);

void solve_extended(IloEnv& env, const TSLP& prob, STAT& stat, IloTimer& clock);

void solve_singlecut(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf);

void solve_level(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option);

bool addToCollection(const VectorXf& dualvec, vector<DualInfo>& dualInfoCollection);

void externalEval(IloEnv& env, Subprob& subp, const TSLP& prob, const IloNumArray& xvals, int sampleSize, vector<double>& objValVec);

void solve_adaptive(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, int option, bool saaError);

void solve_adaptive_partition(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, int option);

void addInitialCuts(IloEnv& env, TSLP& prob, IloModel& mastermodel, const IloNumVarArray& x, const IloNumVarArray& theta, IloCplex& mastercplex, IloRangeArray& cutcon, const vector<int>& samplesForSol, const vector<DualInfo>& dualInfoCollection, const VectorXf& xiterateXf, const vector<VectorXf>& rhsvecs);

bool solve_partly_inexact_bundle(IloEnv& env, TSLP& prob, Subprob& subp, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, int option, bool initial, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int& nearOptimal, double remaintime);

void setup_bundle_QP(IloEnv& env, const TSLP& prob, IloCplex& cplex, IloModel& model, IloNumVarArray& x, const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons);

double coarse_oracle(IloEnv& env, TSLP& prob, Subprob& subp, vector<Component>& partition, IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, IloCplex& cplex, IloModel& model, const IloNumVarArray& x, STAT& stat, IloRangeArray& center_cons, const IloNumArray& stab_center, IloRangeArray& cuts, const vector<double>& cutrhs, VectorXf& aggrCoarseCut, double& coarseCutRhs, vector<VectorXf>& partcoef, vector<double>& partrhs, double starttime, IloTimer& timer, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);

void setAggregatedBounds(const TSLP& prob, const vector<Component>& partition, IloNumArray& secvarlb, IloNumArray& secvarub, IloNumArray& secconstrbd);

double subprob_partition(Subprob& subp, IloNumArray& secvarlb, IloNumArray& secvarub, const TSLP& prob, const IloNumArray& xvals, IloNumArray& duals, const vector<Component>& partition, int k, bool& feasflag);

bool solve_scen_subprobs(IloEnv& env, IloEnv& env2, const TSLP& prob, Subprob& subp, const vector<Component>& partition, const IloNumArray& xvals, double& feasboundscen, VectorXf& cutcoefscen, IloModel& model, const IloNumVarArray& x, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<double>& scenObjs, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);

bool solve_scen_subprobs_target(IloEnv& env, IloEnv& env2, const TSLP& prob, Subprob& subp, const vector<Component>& partition, const IloNumArray& xvals, IloModel& model, const IloNumVarArray& x, vector<Component>& new_partition, STAT& stat, IloTimer& clock, vector<VectorXf>& partcoef, vector<double>& partrhs, double descent_target, bool& fullupdateflag, double& coarseLB, VectorXf& aggrCoarseCut, vector<double>& scenObjs, const vector<int>& samples, vector<DualInfo>& dualInfoCollection, const vector<VectorXf>& rhsvecs, int option);

void add_feas_cuts(IloEnv& env, TSLP& prob, const vector<Component>& partition, IloModel& model, const IloNumVarArray& x, const IloNumArray& xvals, double subobjval, const VectorXf& dualvec, int i);

void computeSamplingError(double& samplingError, const vector<double>& scenObjs);

double solve_warmstart(IloEnv& env, const TSLP& prob, const vector<int>& samples, const IloNumArray& stab_center, const vector<DualInfo>& dualInfoCollection, vector< vector<double> >& cutcoefs, vector<double>& cutrhs, vector<Component>& partition, const vector<VectorXf>& rhsvecs, IloNumArray& xvals, IloTimer & clock, STAT& stat);

void SRP(IloEnv& env, TSLP& prob, Subprob& subp, IloTimer& clock, int nbIterEvalScens, const vector<int>& samplesForEval, double& G, double& S, const IloNumArray& xvals, const IloNumArray& xvals2, vector<double>& scenObjEval);

void BMschedule(Sequence& seq);

void sequentialSetup(Sequence& seq, int option);

void finalEval(IloEnv& env, TSLP& prob, Subprob& subp, const VectorXf& xiterateXf, STAT& stat);
