from ddmtolab.Methods.batch_experiment import BatchExperiment
from ddmtolab.Methods.data_analysis import DataAnalyzer
from ddmtolab.Problems.MTSO.cec17_mtso_10d import CEC17MTSO_10D
from ddmtolab.Algorithms.STSO.BO import BO
from ddmtolab.Algorithms.MTSO.MTBO import MTBO
from ddmtolab.Algorithms.MTSO.MFEA import MFEA

if __name__ == '__main__':
    # # Initialize batch experiment
    # batch_exp = BatchExperiment(base_path='./Data', clear_folder=False)
    #
    # # Load CEC'17 MTSO benchmark problems
    # problems = CEC17MTSO_10D()
    # batch_exp.add_problem(problems.P1, 'P1')
    # batch_exp.add_problem(problems.P2, 'P2')
    # batch_exp.add_problem(problems.P3, 'P3')
    #
    # # Add algorithms: single-task baseline and multi-task variants
    # # batch_exp.add_algorithm(BO, 'BO', n_initial=10, max_nfes=50)
    # # batch_exp.add_algorithm(MTBO, 'MTBO', n_initial=10, max_nfes=50)
    # # batch_exp.add_algorithm(BO_LCB_BCKT, 'BO-LCB-BCKT', n_initial=10, max_nfes=50)
    # # batch_exp.add_algorithm(MFEA, 'MFEA', n=10, max_nfes=50)
    #
    # # Execute experiments with 30 independent runs
    # batch_exp.run(n_runs=10, verbose=True, max_workers=6)

    # Configure data analyzer
    analyzer = DataAnalyzer(
        algorithm_order=['BO', 'MTBO'],
        table_format='latex',
        figure_format='pdf',
        rank_sum_test=True,
        log_scale=True,
        best_so_far=True
    )

    # Generate analysis results
    results = analyzer.run()
