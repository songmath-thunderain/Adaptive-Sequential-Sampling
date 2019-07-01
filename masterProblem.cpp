/*
  Master problem of optimization.
  Interacts with subproblems to improve approximations.
*/

#include "masterProblem.h"

  /*
    Constructor
  */
  masterProblem::masterProblem(/* input variables at least need option */) {
    // *** only solve_level
    meanxvals(meanenv);
    meanobj = solve_mean_value_model(TSLP& prob, IloEnv meanenv, IloNumArray meanxvals, const vector<int>& samples);
    meanxvals.end();
    meanobj.end();
      // quadratic master
      levelmodel(lenv);
      lx(lenv, prob.firstvarlb, prob.firstvarub);
      ltheta(lenv, -IloInfinity, IloInfinity);
    // ***

    mastermodel(env);
    x(env, prob.firstvarlb, prob.firstvarub);
    theta(env, 0, IloInfinity);
  }

  /*
    Destructor
  */
  masterProblem::~masterProblem() {
    mastercplex.end();
    mastermodel.end();
    // *** quadratic (only in solve_level)
    levelcplex.end();
  	levelmodel.end();
  	lx.end();
    // ***
    x.end();
  }

  /*
    Set initial constraints for optimization problem.
  */
  void first_stage_constraints() {
    for (int i = 0; i < prob.firstconstrind.getSize(); ++i)
  	{
      mastermodel.add(find_constraint(i));
  	}
  }

  /*
    Set initial constraint in quadratic master problem.
  */
  void define_first_level_constraint() {
    IloExpr lsum(lenv);
    for (int j = 0; j < prob.nbFirstVars; ++j)
  		lsum += lx[j] * prob.objcoef[j];
  	lsum += ltheta;
  	IloRange rangeub(lenv, -IloInfinity, lsum, IloInfinity);
  	levelmodel.add(rangeub);
  	lsum.end();

    IloObjective lobj = IloMinimize(lenv);
    levelmodel.add(lobj);

    levelcplex(levelmodel);
    levelcplex.setParam(IloCplex::TiLim, 10800);
    levelcplex.setParam(IloCplex::Threads, 1);
    levelcplex.setParam(IloCplex::BarDisplay, 0);
    levelcplex.setOut(env.getNullStream());
  }

  /*
    Helper function for setting initial constraints.
  */
  IloRange find_constraint(int i) {
    IloExpr lhs(env);
    for (int j = 0; j < prob.firstconstrind[i].getSize(); ++j)
      lhs += x[prob.firstconstrind[i][j]]*prob.firstconstrcoef[i][j];
    IloRange range(env, prob.firstconstrlb[i], lhs, prob.firstconstrub[i]);
    lhs.end();
    return range;
  }

  /*

  */
  void add_objective() {
    // Adding objective
  	IloExpr obj(env);
  	for (int i = 0; i < prob.nbFirstVars; ++i)
  		obj += x[i]*prob.objcoef[i];
  	// single cut
  	obj += theta;
  	mastermodel.add(IloMinimize(env, obj));
  	obj.end();
  	mastercplex(mastermodel);
  	mastercplex.setParam(IloCplex::TiLim,10800);
  	mastercplex.setParam(IloCplex::Threads, 1);
  	// Barrier
  	mastercplex.setParam(IloCplex::RootAlg, 2);
  	mastercplex.setParam(IloCplex::BarDisplay, 0);
  	mastercplex.setParam(IloCplex::SimDisplay, 0);
  	mastercplex.setOut(env.getNullStream());
  	//mastercplex.setParam(IloCplex::EpOpt, 1e-4);
  }
