{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ objname }}
{{ underline }}

.. automodule:: {{ fullname }}

 {% block functions %}
 {% if functions %}
 .. rubric:: Functions

 .. autosummary::
    :toctree: stubs
    {% for item in functions %}
    {{ item }}
    {%- endfor %}
 {% endif %}
 {% endblock %}

 {% block classes %}
 {% if classes %}
 .. rubric:: Classes

 .. autosummary::
    :toctree: stubs
    {% for item in classes %}
    {{ item }}
    {%- endfor %}
 {% endif %}
 {% endblock %}

 {% block exceptions %}
 {% if exceptions %}
 .. rubric:: Exceptions

 .. autosummary::
    :toctree: stubs
    {% for item in exceptions %}
    {{ item }}
    {%- endfor %}
 {% endif %}
 {% endblock %}
