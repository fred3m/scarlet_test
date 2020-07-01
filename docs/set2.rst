Test Dataset 2
--------------

.. plot::

    from scarlet_test.measure import all_metrics
    for title, metric in all_metrics.items():
        metric.plot("set2")
        plt.show()
