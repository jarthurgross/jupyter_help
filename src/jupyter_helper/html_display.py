import itertools as it
from IPython.display import display, HTML
import sympy as sy

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
            html += '<td>' + format(entry, format_spec) + '</td>'
        html += '</tr>'
    if format_specs is None:
        format_specs = it.repeat('')
    for row in rows:
        html += '<tr>'
        for entry, format_spec in zip(row, format_specs):
            html += '<td>' + format(entry, format_spec) + '</td>'
        html += '</td>'
    html += '</table>'
    return html

def display_table(rows, format_specs=None, header_entries=None, header_format_specs=None):
    display(HTML(table_format(rows, format_specs, header_entries, header_format_specs)))

def display_align(rows):
    html = r'\begin{align}'
    if isinstance(rows, dict):
        for label, expression in rows.items():
            html += r'{}&={}\\'.format(label, sy.latex(expression).replace(r'\dag', r'\dagger'))
    else:
        for label, expression in rows:
            html += r'{}&={}\\'.format(label, sy.latex(expression).replace(r'\dag', r'\dagger'))
    html += r'\end{align}'
    display(HTML(html))
