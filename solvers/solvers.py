from solvers.admm import ADMMSolver
from solvers.superadmm import SuperADMMSolver
from solvers.superadmmdual import SuperADMMDualSolver
from solvers.superadmm_tricks import SuperADMMTricksSolver
from solvers.ecos import ECOSSolver
from solvers.gurobi import GUROBISolver
from solvers.mosek import MOSEKSolver
from solvers.osqp import OSQPSolver
from solvers.osqppurepy import OSQP as OSQPPythonSolver
from learned_osqp.neural_osqp_solver import NeuralOSQPSolver
# from solvers.qpoases import qpOASESSolver

ECOS = 'ECOS'
ECOS_high = ECOS + "_high"
GUROBI = 'GUROBI'
GUROBI_high = GUROBI + "_high"
OSQP = 'OSQP'
OSQP_high = OSQP + '_high'
OSQP_polish = OSQP + '_polish'
OSQP_polish_high = OSQP_polish + '_high'
MOSEK = 'MOSEK'
MOSEK_high = MOSEK + "_high"
qpOASES = 'qpOASES'
ADMM = 'ADMM'
ADMM_high = ADMM + '_high'
SuperADMM = 'Super'
SuperADMM_high = SuperADMM + '_high'

SuperADMMDual = 'SuperADMMDual'
SuperADMMDual_high = SuperADMMDual + '_high'

Super_ruiz = 'Super_ruiz'
Super_ruiz_high = Super_ruiz + '_high'
Super_ruiz_kaczmarz = 'Super_ruiz_kaczmarz'
Super_ruiz_kaczmarz_high = Super_ruiz_kaczmarz + '_high'
Super_ruiz_ldlt = 'Super_ruiz_ldlt'
Super_ruiz_ldlt_high = Super_ruiz_ldlt + '_high'
Super_ruiz_new_fact = 'Super_ruiz_new_fact'
Super_ruiz_new_fact_high = Super_ruiz_new_fact + '_high'
Super_ruiz_cg = 'Super_ruiz_cg'
Super_ruiz_cg_high = Super_ruiz_cg + '_high'
Super_ruiz_cg_precond = 'Super_ruiz_cg_precond'
Super_ruiz_cg_precond_high = Super_ruiz_cg_precond + '_high'

Super_ldlt = 'Super_ldlt'
Super_ldlt_high = Super_ldlt + '_high'
Super_new_fact = 'Super_new_fact'
Super_new_fact_high = Super_new_fact + '_high'
Super_cg = 'Super_cg'
Super_cg_high = Super_cg + '_high'
Super_cg_precond = 'Super_cg_precond'
Super_cg_precond_high = Super_cg_precond + '_high'

OSQP_python = 'OSQP_python'
OSQP_python_high = OSQP_python + '_high'
OSQP_python_neural = 'OSQP_python_neural'

# solvers = [ECOSSolver, GUROBISolver, MOSEKSolver, OSQPSolver]
# SOLVER_MAP = {solver.name(): solver for solver in solvers}

SOLVER_MAP = {OSQP: OSQPSolver,
              OSQP_high: OSQPSolver,
              OSQP_polish: OSQPSolver,
              OSQP_polish_high: OSQPSolver,
              GUROBI: GUROBISolver,
              GUROBI_high: GUROBISolver,
              MOSEK: MOSEKSolver,
              MOSEK_high: MOSEKSolver,
              ECOS: ECOSSolver,
              ECOS_high: ECOSSolver,
              # qpOASES: qpOASESSolver,
              ADMM: ADMMSolver,
              ADMM_high: ADMMSolver,
              SuperADMM: SuperADMMSolver,
              SuperADMM_high: SuperADMMSolver,
              SuperADMMDual: SuperADMMDualSolver,
              SuperADMMDual_high: SuperADMMDualSolver,
              Super_ruiz: SuperADMMTricksSolver,
              Super_ruiz_high: SuperADMMTricksSolver,
              Super_ruiz_kaczmarz: SuperADMMTricksSolver,
              Super_ruiz_kaczmarz_high: SuperADMMTricksSolver,
              Super_ruiz_ldlt: SuperADMMTricksSolver,
              Super_ruiz_ldlt_high: SuperADMMTricksSolver,
              Super_ruiz_new_fact: SuperADMMTricksSolver,
              Super_ruiz_new_fact_high: SuperADMMTricksSolver,
              Super_ruiz_cg: SuperADMMTricksSolver,
              Super_ruiz_cg_high: SuperADMMTricksSolver,
              Super_ruiz_cg_precond: SuperADMMTricksSolver,
              Super_ruiz_cg_precond_high: SuperADMMTricksSolver,
              Super_ldlt: SuperADMMTricksSolver,
              Super_ldlt_high: SuperADMMTricksSolver,
              Super_new_fact: SuperADMMTricksSolver,
              Super_new_fact_high: SuperADMMTricksSolver,
              Super_cg: SuperADMMTricksSolver,
              Super_cg_high: SuperADMMTricksSolver,
              Super_cg_precond: SuperADMMTricksSolver,
              Super_cg_precond_high: SuperADMMTricksSolver,
              OSQP_python: OSQPPythonSolver,
              OSQP_python_high: OSQPPythonSolver,
              OSQP_python_neural: NeuralOSQPSolver,
              }

time_limit = 1000. # Seconds
eps_low = 1e-03
eps_high = 1e-05

# Solver settings
settings = {
    OSQP: {'eps_abs': eps_low,
           'eps_rel': 0.0,
           'polish': False,
           'max_iter': int(1e09),
           'eps_prim_inf': 1e-15,  # Disable infeas check
           'eps_dual_inf': 1e-15
    },
    OSQP_high: {'eps_abs': eps_high,
                'eps_rel': 0.0,
                'polish': False,
                'max_iter': int(1e09),
                'eps_prim_inf': 1e-15,  # Disable infeas check
                'eps_dual_inf': 1e-15
    },
    OSQP_polish: {'eps_abs': eps_low,
                  'eps_rel': 0.0,
                  'polish': True,
                  'max_iter': int(1e09),
                  'eps_prim_inf': 1e-15,  # Disable infeas check
                  'eps_dual_inf': 1e-15
    },
    OSQP_polish_high: {'eps_abs': eps_high,
                       'eps_rel': 0.0,
                       'polish': True,
                       'max_iter': int(1e09),
                       'eps_prim_inf': 1e-15,  # Disable infeas check
                       'eps_dual_inf': 1e-15
    },
    GUROBI: {'TimeLimit': time_limit,
             'FeasibilityTol': eps_low,
             'OptimalityTol': eps_low,
             },
    GUROBI_high: {'TimeLimit': time_limit,
                  'FeasibilityTol': eps_high,
                  'OptimalityTol': eps_high,
                  },
    MOSEK: {'MSK_DPAR_OPTIMIZER_MAX_TIME': time_limit,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': eps_low,   # Primal feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': eps_low,   # Dual feasibility tolerance
           },
    MOSEK_high: {'MSK_DPAR_OPTIMIZER_MAX_TIME': time_limit,
                 'MSK_DPAR_INTPNT_CO_TOL_PFEAS': eps_high,   # Primal feasibility tolerance
                 'MSK_DPAR_INTPNT_CO_TOL_DFEAS': eps_high,   # Dual feasibility tolerance
                },
    ECOS: {'abstol': eps_low,
           'reltol': eps_low},
    ECOS_high: {'abstol': eps_high,
                'reltol': eps_high},
    qpOASES: {},
    ADMM: {'rho': 1.0,
           'max_iter': int(1e4),
           'abstol': eps_low,
           'reltol': 0.0},
    ADMM_high: {'rho': 1.0,
                'max_iter': int(1e4),
                'abstol': eps_high,
                'reltol': 0.0},
    SuperADMM: {'rho': 0.1,
                'alpha': 500.0,
                'tau': 0.5,
                'b0': 1e8,
                'sigma': 1e-6,
                'max_iter': int(1e4),
                'abstol': eps_low,
                'reltol': 0.0},
    SuperADMM_high: {'rho': 0.1,
                     'alpha': 500.0,
                     'tau': 0.5,
                     'b0': 1e8,
                     'sigma': 1e-6,
                     'max_iter': int(1e4),
                     'abstol': eps_high,
                     'reltol': 0.0},
    SuperADMMDual: {'rho': 1.0,
                    'alpha': 0.8,
                    'max_iter': int(1e4),
                    'abstol': eps_low,
                    'reltol': 0.0
                   },
    SuperADMMDual_high: {'rho': 1.0,
                         'alpha': 0.8,
                         'max_iter': int(1e4),
                         'abstol': eps_high,
                         'reltol': 0.0
                        },
    Super_ruiz: {'rho': 1.0,
                 'alpha': 7.5,
                 'tau': 0.5,
                 'b0': 1e3,
                 'sigma': 1e-6,
                 'max_iter': int(1e4),
                 'abstol': eps_low,
                 'reltol': 0.0,
                 'use_preconditioning': True,
                 'solving_method': 'direct',
                 'max_iter_kacz': 5
              },
    Super_ruiz_high: {'rho': 1.0,
                          'alpha': 7.5,
                          'tau': 0.5,
                          'b0': 1e3,
                          'sigma': 1e-6,
                          'max_iter': int(1e4),
                          'abstol': eps_high,
                          'reltol': 0.0,
                          'use_preconditioning': True,
                          'solving_method': 'direct',
                          'max_iter_kacz': 5
                         },
    Super_ruiz_kaczmarz: {'rho': 1.0,
                 'alpha': 7.5,
                 'tau': 0.5,
                 'b0': 1e3,
                 'sigma': 1e-6,
                 'max_iter': int(1e4),
                 'abstol': eps_low,
                 'reltol': 0.0,
                 'use_preconditioning': True,
                 'solving_method': 'kaczmarz',
                 'max_iter_kacz': 5
              },
    Super_ruiz_kaczmarz_high: {'rho': 1.0,
                              'alpha': 7.5,
                              'tau': 0.5,
                              'b0': 1e3,
                              'sigma': 1e-6,
                              'max_iter': int(1e4),
                              'abstol': eps_high,
                              'reltol': 0.0,
                              'use_preconditioning': True,
                              'solving_method': 'kaczmarz',
                              'max_iter_kacz': 5
                             },
    Super_ruiz_ldlt: {'rho': 1.0,
                 'alpha': 7.5,
                 'tau': 0.5,
                 'b0': 1e3,
                 'sigma': 1e-6,
                 'max_iter': int(1e4),
                 'abstol': eps_low,
                 'reltol': 0.0,
                 'use_preconditioning': True,
                 'solving_method': 'LDLT',
                 'max_iter_kacz': 5
              },
    Super_ruiz_ldlt_high: {'rho': 1.0,
                              'alpha': 7.5,
                              'tau': 0.5,
                              'b0': 1e3,
                              'sigma': 1e-6,
                              'max_iter': int(1e4),
                              'abstol': eps_high,
                              'reltol': 0.0,
                              'use_preconditioning': True,
                              'solving_method': 'LDLT',
                              'max_iter_kacz': 5
                             },
    Super_ruiz_new_fact: {'rho': 1.0,
                       'alpha': 7.5,
                       'tau': 0.5,
                       'b0': 1e3,
                       'sigma': 1e-6,
                       'max_iter': int(1e4),
                       'abstol': eps_low,
                       'reltol': 0.0,
                       'use_preconditioning': True,
                       'solving_method': 'new_fact',
                       'max_iter_kacz': 5
                },
    Super_ruiz_new_fact_high: {'rho': 1.0,
                                     'alpha': 7.5,
                                     'tau': 0.5,
                                     'b0': 1e3,
                                     'sigma': 1e-6,
                                     'max_iter': int(1e4),
                                     'abstol': eps_high,
                                     'reltol': 0.0,
                                     'use_preconditioning': True,
                                     'solving_method': 'new_fact',
                                     'max_iter_kacz': 5
                                    },
    Super_ruiz_cg: {'rho': 1.0,
                       'alpha': 7.5,
                       'tau': 0.5,
                       'b0': 1e3,
                       'sigma': 1e-6,
                       'max_iter': int(1e4),
                       'abstol': eps_low,
                       'reltol': 0.0,
                       'use_preconditioning': True,
                       'solving_method': 'CG',
                       'max_iter_kacz': 5
                },
    Super_ruiz_cg_high: {'rho': 1.0,
                                     'alpha': 7.5,
                                     'tau': 0.5,
                                     'b0': 1e3,
                                     'sigma': 1e-6,
                                     'max_iter': int(1e4),
                                     'abstol': eps_high,
                                     'reltol': 0.0,
                                     'use_preconditioning': True,
                                     'solving_method': 'CG',
                                     'max_iter_kacz': 5
                                    },
    Super_ruiz_cg_precond: {'rho': 1.0,
                       'alpha': 7.5,
                       'tau': 0.5,
                       'b0': 1e3,
                       'sigma': 1e-6,
                       'max_iter': int(1e4),
                       'abstol': eps_low,
                       'reltol': 0.0,
                       'use_preconditioning': True,
                       'solving_method': 'CG_precond',
                       'max_iter_kacz': 5
                },
    Super_ruiz_cg_precond_high: {'rho': 1.0,
                                     'alpha': 7.5,
                                     'tau': 0.5,
                                     'b0': 1e3,
                                     'sigma': 1e-6,
                                     'max_iter': int(1e4),
                                     'abstol': eps_high,
                                     'reltol': 0.0,
                                     'use_preconditioning': True,
                                     'solving_method': 'CG_precond',
                                     'max_iter_kacz': 5
                                    },
    Super_ldlt: {'rho': 1.0,
                 'alpha': 500.0,
                 'tau': 0.5,
                 'b0': 1e8,
                 'sigma': 1e-6,
                 'max_iter': int(1e4),
                 'abstol': eps_low,
                 'reltol': 0.0,
                 'use_preconditioning': True,
                 'solving_method': 'LDLT',
                 'max_iter_kacz': 5
    },
    Super_ldlt_high: {'rho': 1.0,
                 'alpha': 500.0,
                 'tau': 0.5,
                 'b0': 1e8,
                 'sigma': 1e-6,
                 'max_iter': int(1e4),
                 'abstol': eps_high,
                 'reltol': 0.0,
                 'use_preconditioning': True,
                 'solving_method': 'LDLT',
                 'max_iter_kacz': 5
    },
       Super_new_fact: {'rho': 1.0,
                      'alpha': 500.0,
                      'tau': 0.5,
                      'b0': 1e8,
                      'sigma': 1e-6,
                      'max_iter': int(1e4),
                      'abstol': eps_low,
                      'reltol': 0.0,
                      'use_preconditioning': True,
                      'solving_method': 'new_fact',
                      'max_iter_kacz': 5},
       Super_new_fact_high: {'rho': 1.0,
                             'alpha': 500.0,
                             'tau': 0.5,
                             'b0': 1e8,
                             'sigma': 1e-6,
                             'max_iter': int(1e4),
                             'abstol': eps_high,
                             'reltol': 0.0,
                             'use_preconditioning': True,
                             'solving_method': 'new_fact',
                             'max_iter_kacz': 5},
       Super_cg: {'rho': 1.0,
                  'alpha': 500.0,
                  'tau': 0.5,
                  'b0': 1e8,
                  'sigma': 1e-6,
                  'max_iter': int(1e4),
                  'abstol': eps_low,
                  'reltol': 0.0,
                  'use_preconditioning': True,
                  'solving_method': 'CG',
                  'max_iter_kacz': 5},
       Super_cg_high: {'rho': 1.0,
                         'alpha': 500.0,
                         'tau': 0.5,
                         'b0': 1e8,
                         'sigma': 1e-6,
                         'max_iter': int(1e4),
                         'abstol': eps_high,
                         'reltol': 0.0,
                         'use_preconditioning': True,
                         'solving_method': 'CG',
                         'max_iter_kacz': 5},
       Super_cg_precond: {'rho': 1.0,
                  'alpha': 500.0,
                  'tau': 0.5,
                  'b0': 1e8,
                  'sigma': 1e-6,
                  'max_iter': int(1e4),
                  'abstol': eps_low,
                  'reltol': 0.0,
                  'use_preconditioning': True,
                  'solving_method': 'CG_precond',
                  'max_iter_kacz': 5},
       Super_cg_precond_high: {'rho': 1.0,
                         'alpha': 500.0,
                         'tau': 0.5,
                         'b0': 1e8,
                         'sigma': 1e-6,
                         'max_iter': int(1e4),
                         'abstol': eps_high,
                         'reltol': 0.0,
                         'use_preconditioning': True,
                         'solving_method': 'CG_precond',
                         'max_iter_kacz': 5},
       
       OSQP_python:{
             'max_iter': int(1e4),
             'eps_abs': eps_low,
             'eps_rel': eps_low,
       #       'adaptive_rho': False,
             'polish': False,
             'verbose': False,
             'eps_prim_inf': 1e-15,  # Disable infeas check
             'eps_dual_inf': 1e-15
       },
       OSQP_python_high:{
             'max_iter': int(1e4),
             'eps_abs': eps_high,
             'eps_rel': eps_high,
             'polish': False,
             'verbose': False,
             'eps_prim_inf': 1e-15,  # Disable infeas check
             'eps_dual_inf': 1e-15,
             'warm_start': False
       },
       OSQP_python_neural: {
             'max_iter': int(1e4),
             'eps_abs': eps_low,
             'eps_rel': eps_low,
             'polish': False,
             'verbose': False,
             'eps_prim_inf': 1e-15,  # Disable infeas check
             'eps_dual_inf': 1e-15,
       }
}

for key in settings:
    settings[key]['verbose'] = False
    settings[key]['time_limit'] = time_limit
