.PHONY: report presentation clean-report

report: report/report.pdf

report/report.pdf: report/report.tex result/benchmark_results.csv result/plots/mu.png \
	report/generated/iterations_by_p.dat report/generated/condition_by_p.dat \
	report/generated/iterations_by_ref.dat report/generated/speedup_by_np.dat
	latexmk -cd -pdf -interaction=nonstopmode -halt-on-error report/report.tex

report/generated/iterations_by_p.dat report/generated/condition_by_p.dat \
report/generated/iterations_by_ref.dat report/generated/speedup_by_np.dat: \
	result/benchmark_results.csv report/scripts/report_data.py
	python3 report/scripts/report_data.py

clean-report:
	latexmk -cd -C report/report.tex

presentation: heterogeneous_diffusion_presentation.pptx

heterogeneous_diffusion_presentation.pptx: scripts/build_presentation.py \
	result/benchmark_results.csv result/plots/mu.png
	python3 scripts/build_presentation.py
