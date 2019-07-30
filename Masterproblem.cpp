/*
  Master problem of optimization.
  Interacts with subproblems to improve approximations.
*/

#include "Masterproblem.h"

/*
	Default constructor
*/
Masterproblem::Masterproblem() {

}

/*
	Constructor for level master problem
*/
Masterproblem::Masterproblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf) {
	this->env = env;
	this->prob = prob;
	this->stat = stat;
	//this->clock = clock;
	this->samples = samples;
	
	model = IloModel(env);
	x = IloNumVarArray(env, prob.firstvarlb, prob.firstvarub);
	theta = IloNumVar(env, 0, IloInfinity);
	cplex = IloCplex(model);
}

/*
	Constructor for Quadratic Master Problem.
*/
  Masterproblem::Masterproblem(IloEnv& env, TSLP& prob, STAT& stat, IloTimer& clock, const vector<int>& samples, VectorXf& xiterateXf, IloEnv lenv)
	  : Masterproblem::Masterproblem(env, prob, stat, clock, samples, xiterateXf) {
	  // Are all initializations for the LP model done?
	  this->lenv = lenv;
	  levelmodel = IloModel(lenv);
	  lx = IloNumVarArray(lenv, prob.firstvarlb, prob.firstvarub);
	  ltheta = IloNumVar(lenv, -IloInfinity, IloInfinity);
	  lsum = IloExpr(lenv);
	  for (int j = 0; j < prob.nbFirstVars; ++j)
		  lsum += lx[j] * prob.objcoef[j];
	  lsum += ltheta;
	  rangeub = IloRange(lenv, -IloInfinity, lsum, IloInfinity);
	  levelmodel.add(rangeub);
	  lsum.end();
	  lobj = IloObjective(IloMinimize(lenv));
	  levelmodel.add(lobj);
	  levelcplex = IloCplex(levelmodel);
  }

  /*
    Destructor
  */
  Masterproblem::~Masterproblem() {
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
  void Masterproblem::define_lp_model() {
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
  void Masterproblem::define_qp_model() {
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
  void Masterproblem::setup_bundle_QP(const IloNumArray& stab_center, IloObjective& QPobj, IloRangeArray& cuts, IloRangeArray& center_cons) {
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

  /*
	Getter for cplex variable.
  */
  IloCplex& Masterproblem::getCplex() {
	  return cplex;
  }

  /*
	Getter for x variable.
  */
  IloNumVarArray& Masterproblem::getX() {
	  return x;
  }

  /*
  Getter for theta variable.
  */
  IloNumVar& Masterproblem::getTheta() {
	  return theta;
  }

  /*
  Getter for model variable.
  */
  IloModel& Masterproblem::getModel() {
	  return model;
  }

  /*
  Getter for ltheta variable.
  */
  IloNumVar& Masterproblem::getLtheta() {
	  return ltheta;
  }

  /*
  Getter for lx variable.
  */
  IloNumVarArray& Masterproblem::getLx() {
	  return lx;
  }

  /*
  Getter for levelmodel variable.
  */
  IloModel& Masterproblem::getLevelmodel() {
	  return levelmodel;
  }

  /*
  Getter for levelcplex variable.
  */
  IloCplex& Masterproblem::getLevelcplex() {
	  return levelcplex;
  }

  /*
  Getter for rangeub variable.
  */
  IloRange& Masterproblem::getRangeub() {
	  return rangeub;
  }

  /*
  Getter for lobj variable.
  */
  IloObjective& Masterproblem::getLobj() {
	  return lobj;
  }

