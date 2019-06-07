#include "Extended.h"

Extended::Extended(IloEnv& extend_env, const TSLP& extend_prob, STAT& extend_stat, IloTimer& extend_clock)
{
  env = extend_env;
  prob = extend_prob;
  stat = extend_stat;
  clock = extend_clock;
}

Extended::~Extended()
{

}

void Extended::solve_extended()
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
    // The reason why we need to have i and ind separately is that, we have some
    // variables that don't have obj coef, i.e. 0 coef
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