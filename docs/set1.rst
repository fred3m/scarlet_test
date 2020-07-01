Test Dataset 1
--------------

.. plot::

    from scarlet_test.measure import all_metrics
    for title, metric in all_metrics.items():
        metric.plot("set1")
        plt.show()
