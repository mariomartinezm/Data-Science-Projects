RMD_FILES  = $(wildcard *.Rmd)
HTML_FILES = $(wildcard *.html)

.PHONY: all clean

all: ch6_fig_6_3.html ch6_fig_6_4.html ch6_fig_6_5.html ch6_fig_6_7.html ch6_fig_6_9.html ch6_fig_6_11.html

ch6_fig_6_%.html: ./ch6_fig_6_%.Rmd
	Rscript -e "rmarkdown::render('$<')"

clean:
	rm -f $(HTML_FILES)
