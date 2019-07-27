/*
  Master problem of optimization.
  Interacts with subproblems to improve approximations.
*/

#include "MasterProblem.h"

/*
	Default constructor
*/
MasterProblem::MasterProblem() {

}

/*
	Constructor for level master problem
*/
MasterProblem::MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf) {
	this->env = env;
	this->prob = prob;
	this->stat = stat;
	//this->clock = clock;
	this->samples = samples;
	this->xiterateXf = xiterateXf;
	
	model = IloModel(env);
	x = IloNumVarArray(env, prob.firstvarlb, prob.firstvarub);
	theta = IloNumVar(env, 0, IloInfinity);
	cplex = IloCplex(model);
}

/*
  Constructor for solve_level.
  MasterProblem::MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, IloEnv meanenv)
	: MasterProblem::MasterProblem(env, prob, stat, clock, samples, xiterateXf) {
	this->meanenv = meanenv;
	meanxvals = IloNumArray(meanenv);
	meanobj = solve_mean_value_model(prob, meanenv, meanxvals, samples);
	xiterate.resize(prob.nbFirstVars);
	for (int j = 0; j < prob.nbFirstVars; ++j)
		xiterate[j] = meanxvals[j];
	meanxvals.end();
	meanenv.end();
  }
  */

  /*
	Constructor for Quadratic Master Problem.
  */
  MasterProblem::MasterProblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, IloEnv meanenv, IloEnv lenv)
	  : MasterProblem::MasterProblem(env, prob, stat, clock, samples, xiterateXf) {
	  this->meanenv = meanenv;
	  meanxvals = IloNumArray(meanenv);
	  meanobj = solve_mean_value_model(prob, meanenv, meanxvals, samples);
	  xiterate.resize(prob.nbFirstVars);
	  for (int j = 0; j < prob.nbFirstVars; ++j)
		  xiterate[j] = meanxvals[j];
	  meanxvals.end();
	  meanenv.end();

	  this->lenv = lenv;
	  levelmodel = IloModel(lenv);
	  lsum = IloExpr(lenv);
	  for (int j = 0; j < prob.nbFirstVars; ++j)
		  lsum += lx[j] * prob.objcoef[j];
	  lsum += ltheta;
	  rangeub = IloRange(lenv, -IloInfinity, lsum, IloInfinity);
	  levelmodel.add(rangeub);
	  lsum.end();
	  lobj = IloObjective(IloMinimize(lenv));
	  levelmodel.add(lobj);
	  lx = IloNumVarArray(lenv, prob.firstvarlb, prob.firstvarub);
	  ltheta = IloNumVar(lenv, -IloInfinity, IloInfinity);
	  levelcplex = IloCplex(levelmodel);
  }

  /*
    Destructor
  */
  MasterProblem::~MasterProblem() {
    cplex.end();
    model.end();
    // *** quadratic (only in solve_level)
    levelcplex.end();
  	levelmodel.end();
  	lx.end();
    // ***
    x.end();
  }
  
  /*
    Set initial constraints for level master problem.
  */
  void MasterProblem::define_lp_model() {
	// Adding first-stage constraints
    for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
  	{
		IloExpr lhs(env);
		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
			lhs += x[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
		model.add(range);
		lhs.end();
  	}
	// Adding objective
	IloExpr obj(env);
	for (int i = 0; i < prob.nbFirstVars; ++i)
		obj += x[i] * prob.objcoef[i];
	// single cut
	obj += theta;
	model.add(IloMinimize(env, obj));
	obj.end();
	cplex = IloCplex(model);
	cplex.setParam(IloCplex::TiLim, 10800);
	cplex.setParam(IloCplex::Threads, 1);
	// Barrier
	cplex.setParam(IloCplex::RootAlg, 2); // Only for lp definition
	cplex.setParam(IloCplex::BarDisplay, 0);
	cplex.setParam(IloCplex::SimDisplay, 0);
	cplex.setOut(env.getNullStream());
	//cplex.setParam(IloCplex::EpOpt, 1e-4);
  }

  /*
    Set initial constraint for quadratic master problem.
  */
  void MasterProblem::define_qp_model() {
	// Adding first-stage constraints
	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
	{
	  IloExpr lhs(lenv);
	  for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
		  lhs += lx[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
	  IloRange range(lenv, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
	  levelmodel.add(range);
	  lhs.end();
	}

    levelcplex = IloCplex(levelmodel);
    levelcplex.setParam(IloCplex::TiLim, 10800);
    levelcplex.setParam(IloCplex::Threads, 1);
    levelcplex.setParam(IloCplex::BarDisplay, 0);
    levelcplex.setOut(env.getNullStream());
  }

  /*
	Set up master problem with partition
  */
  void MasterProblem::setup_bundle_QP(const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons) {
  	// first stage constraints
  	for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
  	{
  		IloExpr lhs(env);
  		for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
  			// Perform refinement
  			lhs += x[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
  		IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
  		model.add(range);
  		lhs.end();
  	}
  	// Adding objective
  	QPobj = IloMinimize(env);
  	model.add(QPobj);
  	IloExpr objExpr(env);
  	IloNumVarArray y(env, prob.nbFirstVars, 0, IloInfinity);
  	for (int j = 0; j < prob.nbFirstVars; ++j)
  		objExpr += y[j];
  	QPobj.setExpr(objExpr);
  	objExpr.end();

  	for (int j = 0; j < prob.nbFirstVars; ++j)
  	{
  		IloRange range(env, -IloInfinity, y[j] - x[j], IloInfinity);
  		model.add(range);
  		center_cons.add(range);
  	}

  	for (int j = 0; j < prob.nbFirstVars; ++j)
  	{
  		IloRange range(env, -IloInfinity, y[j] + x[j], IloInfinity);
  		model.add(range);
  		center_cons.add(range);
  	}

  	// Add cuts
  	for (int l = 0; l < cuts.getSize(); ++l)
  		model.add(cuts[l]);
  	cplex = IloCplex(model);
  	cplex.setParam(IloCplex::TiLim, 3600);
  	cplex.setParam(IloCplex::Threads, 1);
  	cplex.setParam(IloCplex::BarDisplay, 0);
  	cplex.setParam(IloCplex::SimDisplay, 0);
  	cplex.setOut(env.getNullStream());
  	//cplex.setParam(IloCplex::EpOpt, 1e-4);
  }

  double MasterProblem::solve_mean_value_model(const TSLP& prob, IloEnv& meanenv, IloNumArray& meanxvals, const vector<int>& samples) {
    IloModel meanmodel(meanenv);
    IloNumVarArray meanx(meanenv, prob.firstvarlb, prob.firstvarub);
    IloNumVarArray meany(meanenv, prob.nbSecVars, -IloInfinity, IloInfinity);
    // first-stage constraints
    for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
    {
      IloExpr lhs(meanenv);
      for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
        lhs += meanx[prob.firstconstrind[i][j]] * prob.firstconstrcoef[i][j];
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
          lhs += meany[ind - prob.nbFirstVars] * prob.CoefMat[i][j];
        else
          lhs += meanx[ind] * prob.CoefMat[i][j];
      }
      // set constraint bounds
      double bd = 0;
      for (int k = 0; k < nbScens; ++k)
        bd += prob.secondconstrbd[samples[k] * prob.nbSecRows + i];
      bd = bd * 1.0 / nbScens;
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
          bd += prob.secondvarlb[samples[k] * prob.nbSecVars + i];
        bd = bd * 1.0 / nbScens;
        range.setLB(bd);
      }
      if (prob.secondvarub[i] != IloInfinity)
      {
        double bd = 0;
        for (int k = 0; k < nbScens; ++k)
          bd += prob.secondvarub[samples[k] * prob.nbSecVars + i];
        bd = bd * 1.0 / nbScens;
        range.setUB(bd);
      }
      meanmodel.add(range);
      lhs.end();
    }
    IloExpr meanobj(meanenv);
    for (int i = 0; i < prob.nbFirstVars; ++i)
      meanobj += meanx[i] * prob.objcoef[i];

    for (int i = 0; i < prob.objcoef.getSize(); ++i)
    {
      if (i >= prob.nbFirstVars)
        meanobj += meany[i - prob.nbFirstVars] * prob.objcoef[i];
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
	Getter for cplex variable.
  */
  IloCplex& MasterProblem::getCplex() {
	  return cplex;
  }

  /*
	Getter for x variable.
  */
  IloNumVarArray& MasterProblem::getX() {
	  return x;
  }

  /*
  Getter for theta variable.
  */
  IloNumVar& MasterProblem::getTheta() {
	  return theta;
  }

  /*
  Getter for model variable.
  */
  IloModel& MasterProblem::getModel() {
	  return model;
  }

  /*
  Getter for meanxvals variable.
  */
  IloNumArray& MasterProblem::getMeanxvals() {
	  return meanxvals;
  }

  /*
  Getter for xiterate variable.
  */
  double MasterProblem::getXiterateVal(int j) {
	  return xiterate[j];
  }

  /*
  Getter for ltheta variable.
  */
  IloNumVar& MasterProblem::getLtheta() {
	  return ltheta;
  }

  /*
  Getter for lx variable.
  */
  IloNumVarArray& MasterProblem::getLx() {
	  return lx;
  }

  /*
  Getter for levelmodel variable.
  */
  IloModel& MasterProblem::getLevelmodel() {
	  return levelmodel;
  }

  /*
  Getter for levelcplex variable.
  */
  IloCplex& MasterProblem::getLevelcplex() {
	  return levelcplex;
  }

  /*
  Getter for rangeub variable.
  */
  IloRange& MasterProblem::getRangeub() {
	  return rangeub;
  }

  /*
  Getter for lobj variable.
  */
  IloObjective& MasterProblem::getLobj() {
	  return lobj;
  }

  /*
  Setter for xiterate variable.
  */
  void MasterProblem::setXiterateVal(int pos, double num) {
	  xiterate[pos] = num;
  }

  /*
  Setter for xiterateXf variable
  */
  void MasterProblem::setXiterateXF(int pos, double num) {
	  xiterateXf(pos) = num;
  }