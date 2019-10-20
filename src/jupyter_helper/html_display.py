import itertools as it
import re
from IPython.display import display, HTML
import sympy as sy

def sanitize_for_html(text):
    # Adapted from https://stackoverflow.com/a/6117124/1236650
    rep = {'&': '&amp;',
           '<': '&lt;',
           '>': '&gt;'}
    rep = dict((re.escape(k), v) for k, v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    return text

def table_format(rows, format_specs=None, header_entries=None, header_format_specs=None):
    html = '<table>'
    if header_entries is not None:
        if header_format_specs == 'same':
            # Use the same format specifiers the body rows use
            header_format_specs = format_specs
        if header_format_specs is None:
            header_format_specs = it.repeat('')
        html += '<tr>'
        for entry, format_spec in zip(header_entries, header_format_specs):
            html += '<th>' + sanitize_for_html(format(entry, format_spec)) + '</th>'
        html += '</tr>'
    if format_specs is None:
        format_specs = it.repeat('')
    for row in rows:
        html += '<tr>'
        for entry, format_spec in zip(row, format_specs):
            html += '<td>' + sanitize_for_html(format(entry, format_spec)) + '</td>'
        html += '</td>'
    html += '</table>'
    return html

def display_table(rows, format_specs=None, header_entries=None, header_format_specs=None):
    display(HTML(table_format(rows, format_specs, header_entries, header_format_specs)))

def display_align(rows):
    '''Display `sympy` expressions as a sequence of equations.

    Each element of the list `rows` should be a pair whose first element is a
    LaTeX expression signifying what the `sympy` object in the second element
    represents.

    '''
    html = r'\begin{align}'
    if isinstance(rows, dict):
        for label, expression in rows.items():
            html += r'{}&={}\\'.format(label, sy.latex(expression).replace(r'\dag', r'\dagger'))
    else:
        for label, expression in rows:
            html += r'{}&={}\\'.format(label, sy.latex(expression).replace(r'\dag', r'\dagger'))
    html += r'\end{align}'
    display(HTML(html))
