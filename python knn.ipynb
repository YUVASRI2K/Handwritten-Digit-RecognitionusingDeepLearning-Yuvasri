{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNee1X1h+2CihQ8RLU6Cw0l",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/YUVASRI2K/Handwritten-Digit-RecognitionusingDeepLearning-Yuvasri/blob/main/knn.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 783
        },
        "id": "-fLMmYSAlqow",
        "outputId": "d3e4abb9-7ba0-4775-f9a5-2a90f5c31de5"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loading training data...\n",
            "Loading testing data...\n",
            "Training images shape: (60000, 784)\n",
            "Training labels shape: (60000,)\n",
            "Testing images shape: (10000, 784)\n",
            "Testing labels shape: (10000,)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaAAAAG0CAYAAAB0cfPUAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAoSklEQVR4nO3df3RU5Z3H8c/kJ8lkYIBsgERJCCGkFggRgV2VgoBLj4u1gCvUY9Uq0HMCLtoi21NaJdTYsgQKrHS3q9ES66qRYxQpiltBGsQtuKEcMVKU8DMYOWAmCRMSEjL7x2xmHTIhucOEJ5O8X+dw4D73fuc++XKTT+6dmTs2j8fjEQAA11iE6QkAAHonAggAYAQBBAAwggACABhBAAEAjCCAAABGEEAAACMIIACAEQQQAMAIAggI0oMPPiibzaZjx4512T5WrFghm82m999/v8v2AZhCAKFHs9lsstlspqfRLbQGZnt/Dh06ZHqK6GWiTE8AwLW1ZMkSOZ3ONuOJiYnXfjLo1QggoJd59NFHlZaWZnoaAJfggFZvvPGG7rvvPmVmZsput8tut2vcuHHasGGDWlpa2q1raWnR2rVrlZWVpT59+ui6667TY489ptra2oDbnzp1SosXL1Z6erpiY2M1cOBAfec739G+ffu66ksDuiXOgID/85Of/EQRERGaOHGiUlJSVFNTox07dmjJkiXat2+fXnzxxYB1jz32mP70pz/pnnvu0V133aXt27dr3bp1Ki0t1e7du9WnTx/ftmVlZfr7v/97ffXVV5oxY4Zmz56ts2fP6o033tCtt96qkpIS3XHHHR3OdcWKFcrLy9OTTz6pFStWWPo63377bdXW1ioyMlIZGRmaOnWq+vbta+kxgJDwAD2YJE9nD/PPP/+8zdilS5c8999/v0eS57//+7/91j3wwAMeSZ6BAwd6jh075lcze/ZsjyTPypUrfeNNTU2e4cOHe2JjYz3vv/++32NVVlZ6kpOTPYMHD/Y0NDT4xp988kmPJM/OnTv9tm8df/LJJzv1tX19vpf/cTgcnmeeeabTjwOECpfggP8zfPjwNmMRERFasmSJJGn79u0B65YsWaLU1FS/mtWrVysiIkLPP/+8b/wPf/iDjhw5okceeUSTJ0/2e4zk5GQtW7ZMVVVVeu+99zqc6+LFi/Xpp59q8eLFnfraJOlb3/qWXn31VR0/flwXLlzQkSNHVFBQ4Hu8//iP/+j0YwGhwCU44P+cO3dOq1ev1rZt21RRUSG32+23vrKyMmDd5WEiSenp6br++ut17NgxuVwuOZ1Offjhh5Kk48ePB7xs9tlnn0mSPv300w4vwyUmJlp+1dpDDz3UZo4//vGPNXLkSN15551avny5Hn74YUVGRlp6XCBYBBAgyeVyafz48Tp69KgmTJig+++/XwMGDFBUVJRcLpfWr1+vxsbGgLWDBg0KOD548GAdP35cNTU1cjqdOnfunCTptddeu+Jczp8/f3VfjEUzZ85USkqKKisrVV5ertGjR1/T/aP3IoAASc8995yOHj0a8En9Dz/8UOvXr2+39ssvv9TIkSPbjFdVVUmS+vXr5/f3m2++qe985zshmnlo/M3f/I0qKyvbnPUBXYnngABJn3/+uSRpzpw5bdbt2rXrirWB1ldUVOjkyZNKS0vzvenzb//2byVJpaWlVznb0KqpqdGhQ4dks9k0bNgw09NBL0IAAZLvjZmX33Nt//79+uUvf3nF2vXr1+v48eO+5ZaWFj3++ONqaWnRD37wA9/4XXfdpeHDh2vjxo3atm1bwMf68MMPVV9f3+F8z549q0OHDuns2bMdbit5z8ZOnTrVZvz8+fN68MEH1dDQoOnTp7d7ORHoClyCQ6/w4IMPtrvuN7/5je6//36tXr1ajz76qHbu3KkRI0bos88+09atWzV79my9+uqr7dbfcsstGjt2rObOnat+/fpp+/btOnDggMaNG6dly5b5touOjtbrr7+uGTNm6B/+4R908803a+zYsYqPj9fJkye1b98+VVRU6IsvvlB8fPwVv55nnnnG0vuADh06pOnTp+vv/u7vlJmZqaSkJFVWVuq//uu/VFVVpfT0dD333HMdPg4QSgQQeoVNmza1u27dunVKTk5WaWmpfvKTn2j37t3avn27srKy9Jvf/EbTp0+/YgD9+te/VklJiZ599lkdO3ZMAwcO1JIlS7Ry5Uq/N6FK0pgxY3TgwAGtXbtWW7du1QsvvKCIiAgNGTJEOTk5ysvL65J7sg0fPlwPP/yw9u3bpy1btsjlcik+Pl4jR47U4sWL9U//9E9yOBwh3y9wJTaPx+MxPQkAQO/Dc0AAACMIIACAEQQQAMAIAggAYAQBBAAwggACABhBAAEAjCCAAABGdNs7Idx77706dOiQ35jdbldpaakmTZrUq+/aSx+86IMXffCiD17doQ9ZWVn6z//8zw6367I7Ibzzzjt666235HK5lJqaqoceekgZGRmdrr/xxhu1f/9+vzGHw6Ha2lr17dtXdXV1oZ5y2KAPXvTBiz540Qev7tCHnJwclZWVdbhdl1yC27Nnj4qKinT33Xdr1apVSk1NVX5+vmpqarpidwCAMNQlAbR161ZNmzZNt912m6677jotWLBAMTEx2rlzZ1fsDgAQhkL+HFBzc7MqKir03e9+1zcWERGh0aNH6/Dhw222b2pqUlNTk2/ZZrMpLi5Odru9zd15W5d7+1176YMXffCiD170was79MFut3dqu5AHUG1trVpaWnyfAtnK6XTq9OnTbbYvKSnR5s2bfcvDhg3TqlWrrvipkZWVlSGbbzijD170wYs+eNEHr3Dog/FXwc2aNUszZ870LdtsNknSpEmTdODAAb9tHQ6HKisrlZKS0uufZKQP9KEVffCiD17doQ/Z2dmd+uj5kAdQ3759FRERIZfL5TfucrnanBVJ3k+JjI6ObjPudrvbbV5dXV2vPsBa0Qcv+uBFH7zog5fJPnT25d8hfxFCVFSU0tPTdfDgQd9YS0uLDh48qMzMzFDvDgAQprrkEtzMmTO1ceNGpaenKyMjQ9u2bVNjY6OmTJnSFbsDAIShLgmgm2++WbW1tSouLpbL5VJaWpp++tOfBrwEBwDonbrsRQjf/va39e1vf7urHh4AEOa4GSkAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGBFlegJAdxIZGWm5pl+/fl0wk85zOBySpAEDBig6Otpv3eLFi4N6zPj4eMs1I0eOtFyzaNEiyzUFBQUBx6OivD/OCgsL1dzc7Lfue9/7nuX9SFJDQ4Plml/96leWa/Ly8izX9AScAQEAjCCAAABGhPwSXHFxsTZv3uw3lpycrHXr1oV6VwCAMNYlzwFdf/31+vnPf+5bjojgRAsA4K9LAigiIkJOp7MrHhoA0EN0SQBVVVXphz/8oaKjo5WZmal7771XiYmJAbdtampSU1OTb9lmsykuLk52u9336p5WrcuXj/c29MGrK/oQzKvgTP8/JCQk+P39dbGxsUE9ZkxMjOWa1lehWRFozsHup3U8mHmEUjA9D+Ux1B1+Ptjt9k5tZ/N4PJ5Q7nj//v1qaGhQcnKyqqurtXnzZn311Vdas2aN4uLi2mx/+XNGw4YN06pVq0I5JQBANxTyALqc2+1Wbm6uHnjgAU2dOrXN+vbOgCZNmqQDBw74betwOFRZWamUlBTV1dV15bS7Nfrg1RV9CMf3ASUkJOjgwYMaNWqUzp8/77du4cKFQT1moF8WOzJixAjLNT/+8Y8t1zz11FMBx6OiojRr1iyVlJS0eR/QP/7jP1rejxTc+4B+/etfW6755S9/abmmPd3h50N2drZKS0s73K7Lz1XtdruSk5NVVVUVcH10dHSbN89J3uBqr3l1dXW9+gdvK/rgFco+BBNA3eVFNufPn2/Th8bGxqAeK5g+XP5DvzMuD8xQ7Ke5uTmouYRKMD3viu9jkz8f3G53p7br8u+choYGVVVV8aIEAICfkJ8BFRUV6aabblJiYqKqq6tVXFysiIgI3XrrraHeFQAgjIU8gL766iutX79edXV16tu3r7KyspSfn6++ffuGelcAgDAW8gB69NFHQ/2Q6KaGDh1quSaYl/fefPPNAcf79OkjyXujycufLA72jDuYS8Vz5swJal+hVlFRYXT/p06dslyzYcMGyzWzZs264vpA/x/BPhdy+QuhOmPXrl1B7as36h7PngIAeh0CCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGGH2w9PRLYwdOzaouh07dliu6YpPD/23f/u3kD9mb9fS0mK55mc/+5nlmmA+kO6ll14KOB4XF6eXXnpJ3//+93XhwgW/dV988YXl/UhSdXW15Zq//vWvQe2rN+IMCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEZwN2zoxIkTQdWdO3fOck1X3A07HP35z3+2XONyuQKOR0VF6fbbb9d7772n5uZmv3W33XZbMNPTxYsXLde8+OKLQe0rVBwOhyTprbfeUl1dndG5oHM4AwIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAI7gZKfTVV18FVff4449brpk5c6blmv379wcc79OnjwoKCrRs2TI1NDT4rduwYYPl/QTrL3/5i+Wa22+/3XKN2+0OOO5wOFRbW6s5c+a0uQnnN7/5Tcv7kaQlS5YEVQdYwRkQAMAIAggAYAQBBAAwggACABhBAAEAjCCAAABGEEAAACMIIACAEQQQAMAIAggAYAQBBAAwggACABjBzUgRtDfeeMNyzY4dOyzXXH6DzVYOh0MFBQV69tln22yTnZ1teT+S9PDDD1uuKSgosFzT3o1FQ+2TTz4Jqm7hwoUhngnQFmdAAAAjCCAAgBGWL8GVl5dry5YtOnr0qKqrq7V06VJNmDDBt97j8ai4uFjvvfee3G63srKyNH/+fA0ZMiSkEwcAhDfLZ0CNjY1KS0tr91r5m2++qbffflsLFizQ008/rdjYWOXn5+vixYtXPVkAQM9hOYBycnI0b948v7OeVh6PR9u2bdPs2bM1fvx4paamavHixaqurta+fftCMmEAQM8Q0lfBnTlzRi6XS2PGjPGNxcfHKyMjQ4cPH9Ytt9zSpqapqUlNTU2+ZZvNpri4ONntdjkcDr9tW5cvH+9twrkPoZzzlfoQHR0dsv10JC4uznLNtepDb0IfvLpDH+x2e6e2C2kAuVwuSVK/fv38xvv16+dbd7mSkhJt3rzZtzxs2DCtWrVKpaWl7e6nsrLyqufaE9AHr1OnThndf2Fh4TWp6QjHgxd98AqHPhh/H9CsWbM0c+ZM37LNZpMkTZo0SQcOHPDb1uFwqLKyUikpKe2+N6Q3COc+9O3b13LNld4HdOrUKV133XVttlm/fn1Q87v//vst1yxYsMByzWuvvWa5pj3hfDyEEn3w6g59yM7OvuJJRKuQBpDT6ZQk1dTUqH///r7xmpoapaWlBayJjo4OeLnE7Xa327y6urpefYC1Csc+tP6CYUVHX2OgPnz9sm5Xu3DhguWarvh/C8fjoSvQBy+TfejsG61D+j6gpKQkOZ1Offzxx76x+vp6ff7558rMzAzlrgAAYc7yGVBDQ4Oqqqp8y2fOnNGxY8eUkJCgxMRE3XHHHXr99dc1ZMgQJSUl6ZVXXlH//v01fvz4kE4cABDeLAfQkSNHlJeX51suKiqSJE2ePFmLFi3SXXfdpcbGRv32t79VfX29srKy9NOf/lQxMTGhmzUAIOxZDqBvfvObKi4ubne9zWbT3LlzNXfu3KuaGHqm2trakD2Wx+Px/d3671Y1NTUh209HgnkRwquvvmq5pqWlxXIN0J1xLzgAgBEEEADACAIIAGAEAQQAMIIAAgAYQQABAIwggAAARhBAAAAjCCAAgBEEEADACAIIAGAEAQQAMIIAAgAYYfwjuYGusGLFiqDqxo0bZ7lm8uTJlmumT59uuebdd9+1XAN0Z5wBAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIAR3IwUPZLb7Q6qbsGCBZZrysrKLNc8++yzlmt27twZcDw6OlqS9O///u9qamryW/fRRx9Z3o8kbdy40XKNx+MJal/ovTgDAgAYQQABAIwggAAARhBAAAAjCCAAgBEEEADACAIIAGAEAQQAMIIAAgAYQQABAIwggAAARhBAAAAjuBkp8DVHjhyxXPPggw9arnnhhRcs13z/+9+/4vp58+ZZrmmP3W63XFNUVGS55osvvrBcg56DMyAAgBEEEADACAIIAGAEAQQAMIIAAgAYQQABAIwggAAARhBAAAAjCCAAgBEEEADACAIIAGAEAQQAMIKbkQJXqaSkxHLNZ599Zrlm7dq1AccjIyM1depUvf/++7p06ZLfumnTplnejyQ9/fTTlmtSU1Mt1+Tn51uuqaystFyD7okzIACAEQQQAMAIy5fgysvLtWXLFh09elTV1dVaunSpJkyY4Fu/ceNG7dq1y68mOztby5cvv/rZAgB6DMsB1NjYqLS0NE2dOlUFBQUBtxk7dqxyc3P/fydRPNUEAPBnORlycnKUk5Nz5QeNipLT6Qx2TgCAXqBLTk3Ky8s1f/582e12jRo1SvPmzZPD4Qi4bVNTk5qamnzLNptNcXFxstvtbWpal9t7rN6CPniFcx+C+cjryMjIK463t/5aiY6OtlyTkJBguaa9/+9wPh5CqTv0obPHt83j8XiC3ck999zT5jmgDz74QLGxsUpKSlJVVZVefvll9enTR/n5+YqIaPuah+LiYm3evNm3PGzYMK1atSrYKQEAwkTIz4BuueUW37+HDh2q1NRUPfLII/rkk080evToNtvPmjVLM2fO9C3bbDZJ0qRJk3TgwAG/bR0OhyorK5WSkqK6urpQTz1s0AevcO7DDTfcYLmmvffmREZGavLkydq1a1eb9wFNmTIlmOkF5fnnn7dc097zyFdy+vTpgOPhfDyEUnfoQ3Z2tkpLSzvcrstfHTBo0CA5HA5VVVUFDKDo6OiAp+5ut7vd5tXV1fXqA6wVffAKxz643W7LNZeHS6D1HW3Tlb5+Kb2zzp8/b7mmo//rcDweuoLJPnT2+O7y9wGdO3dO58+fV//+/bt6VwCAMGL5DKihoUFVVVW+5TNnzujYsWNKSEhQQkKCXnvtNU2cOFFOp1Nffvmlfv/732vw4MHKzs4O6cQBAOHNcgAdOXJEeXl5vuWioiJJ0uTJk7VgwQKdOHFCu3btktvt1oABAzRmzBjNnTs3qFfIAAB6rqt6FVxXuvHGG7V//36/MYfDodraWvXt27dXX+OlD169rQ/tvbfO4XDoxIkTGjp0aJs+3HnnnUHt64UXXrBc0/oCIit27Nhhueb2228PON7bjof2dIc+5OTkqKysrMPtuBccAMAIAggAYAQBBAAwggACABhBAAEAjCCAAABGEEAAACMIIACAEQQQAMAIAggAYAQBBAAwggACABhBAAEAjOjyT0QFEBoulyvgeOunoLpcrjZ3P37xxReD2tdzzz1nuSYqyvqPk29961uWa9r7mPH4+HhJ0qRJk1RfX++37v3337e8H3Q9zoAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAhuRgoYMGbMGMs1d999d8DxmJgYSdLy5ct18eJFv3Xjx4+3PjkFd2PRYJSXl1uu+dOf/hRw3OFwSJI++OCDNjdlRffEGRAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGMHNSIGvGTlypOWaxYsXW66ZPXu25ZrBgwdfcf3jjz9u+TFD6dKlS5ZrvvjiC8s1LS0tVxxvaWlpdxt0L5wBAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIAR3IwU3V57N+FMSEiQJA0aNEh2u91v3fe+972g9hXMjUXT0tKC2ld39tFHH1muyc/Pt1yzZcsWyzXoOTgDAgAYQQABAIywdAmupKREe/fuVWVlpWJiYpSZman77rtPycnJvm0uXryooqIi7dmzR01NTcrOztb8+fPldDpDPXcAQBizdAZUXl6uGTNmKD8/Xz/72c906dIlPfXUU2poaPBts2nTJv3P//yPfvSjHykvL0/V1dVas2ZNyCcOAAhvlgJo+fLlmjJliq6//nqlpaVp0aJFOnv2rCoqKiRJ9fX12rFjhx544AGNGjVK6enpys3N1V//+lcdPny4S74AAEB4uqpXwdXX10v6/1cjVVRU6NKlSxo9erRvm5SUFCUmJurw4cPKzMxs8xhNTU1qamryLdtsNsXFxclut8vhcPht27p8+Xhv09v60Hp8tTceaH1sbGxQ+7LZbEHV9TQREdafHo6Li7NcE8pjuLd9X7SnO/Th8leltifoAGppadHvfvc7jRw5UkOHDpUkuVwuRUVFtdl5v3795HK5Aj5OSUmJNm/e7FseNmyYVq1apdLS0nb3XVlZGey0exT64LV//37TU+gWggmN9tx0002Wa1555ZWQ7f9q8H3hFQ59CDqACgsLdfLkSa1cufKqJjBr1izNnDnTt9z6G+ikSZN04MABv20dDocqKyuVkpKiurq6q9pvOOttfRg0aFDA8YSEBO3fv185OTk6f/6837q77747qH0tXLjQck1qampQ+wqliIgItbS0hOzxysrKLNcUFBRYrtm2bZvlmvb0tu+L9nSHPmRnZ1/xJKJVUAFUWFiosrIy5eXlaeDAgb5xp9Op5uZmud1uv7Ogmpqadl8FFx0drejo6Dbjbre73ebV1dX16gOsVW/pQ0en8+fPn28TQI2NjUHty+PxBFXX0wQTZhcuXLBc0xXHb2/5vuiIyT643e5ObWfpnN3j8aiwsFB79+7VE088oaSkJL/16enpioyM1Mcff+wbO336tM6ePRvw+R8AQO9l6QyosLBQu3fv1rJlyxQXF+d7Xic+Pl4xMTGKj4/X1KlTVVRUpISEBMXHx+v5559XZmYmAQQA8GMpgN59911J0ooVK/zGc3NzNWXKFEnSAw88IJvNpjVr1qi5udn3RlQAAL7OUgAVFxd3uE1MTIzmz59P6PQC7b044EpuuOEGyzXPPPNMwPHWV3299dZbbZ6zyMrKsryf7u7Pf/5zwPHIyEhNmDBBH330kS5duuS3bvXq1UHt680337RcE8oXQaB34F5wAAAjCCAAgBEEEADACAIIAGAEAQQAMIIAAgAYQQABAIwggAAARhBAAAAjCCAAgBEEEADACAIIAGAEAQQAMCLoj+RG9zRgwADLNb/97W+D2tfYsWMt16Snpwe1rysx/VlTe/bssVyzZs0ayzXbt28POO5wOPTll1/qzjvvbPMJmMF8SilwrXAGBAAwggACABhBAAEAjCCAAABGEEAAACMIIACAEQQQAMAIAggAYAQBBAAwggACABhBAAEAjCCAAABGcDPSa2TixImWax5//PGA41FR3v+2F198Uc3NzX7rJkyYYHk/KSkplmu6u/r6+qDqNmzYYLnm6aeftlzjdrst17Sn9Xi4cOECNx9FWOEMCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCM4Gak18isWbNCXnPnnXcGO52QKC8vt1yzdetWyzWX33C1VUxMjJYtW6aCggJdvHjRb92aNWss70eSXC5XUHUArOMMCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQDM8HRTOTk5Hkl+fxwOh8fj8XgcDkebdb3pD32gD/SBPnTnPuTk5HTq5zxnQAAAIwggAIARlj4PqKSkRHv37lVlZaViYmKUmZmp++67T8nJyb5tVqxY0eZzYqZPn66FCxeGZsYAgB7BUgCVl5drxowZGj58uC5duqSXX35ZTz31lNauXas+ffr4tps2bZrmzp3rW46JiQndjAEAPYKlAFq+fLnf8qJFizR//nxVVFTohhtu8I3HxsbK6XSGZIIAgJ7pqj6Su76+XpKUkJDgN15aWqrS0lI5nU6NGzdOc+bMUWxsbMDHaGpqUlNTk2/ZZrMpLi5OdrtdDofDb9vW5cvHexv64EUfvOiDF33w6g59sNvtndrO5vF4PMHsoKWlRf/yL/8it9utX/ziF77xP/7xj0pMTNSAAQN0/PhxvfTSS8rIyNDSpUsDPk5xcbE2b97sWx42bJhWrVoVzJQAAGEk6AB69tln9Ze//EUrV67UwIED293u4MGDWrlypTZs2KDBgwe3Wd/eGdCkSZN04MABv20dDocqKyuVkpKiurq6YKbdI9AHL/rgRR+86INXd+hDdna2SktLO9wuqEtwhYWFKisrU15e3hXDR5IyMjIkSVVVVQEDKDo6WtHR0W3G3W53u82rq6vr1QdYK/rgRR+86IMXffAy2Qe3292p7Sy9D8jj8aiwsFB79+7VE088oaSkpA5rjh07Jknq37+/lV0BAHo4S2dAhYWF2r17t5YtW6a4uDi5XC5JUnx8vGJiYlRVVaXdu3frxhtvVEJCgk6cOKFNmzbpG9/4hlJTU7ti/gCAMGUpgN59911J3jebfl1ubq6mTJmiqKgoffzxx9q2bZsaGxs1cOBATZw4UbNnzw7ZhAEAPYOlACouLr7i+sTEROXl5V3VhAAAvQP3ggMAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGBFlegLtycrKajNmt9slSdnZ2XK73dd6St0GffCiD170wYs+eHWHPgT6+R2IzePxeLp4LgAAtBFWl+AuXLigf/7nf9aFCxdMT8Uo+uBFH7zogxd98AqnPoRVAHk8Hh09elS9/aSNPnjRBy/64EUfvMKpD2EVQACAnoMAAgAYEVYBFB0drbvvvlvR0dGmp2IUffCiD170wYs+eIVTH3gVHADAiLA6AwIA9BwEEADACAIIAGAEAQQAMIIAAgAY0W1vRnq5d955R2+99ZZcLpdSU1P10EMPKSMjw/S0rqni4mJt3rzZbyw5OVnr1q0zM6FrpLy8XFu2bNHRo0dVXV2tpUuXasKECb71Ho9HxcXFeu+99+R2u5WVlaX58+dryJAhBmcdeh31YePGjdq1a5dfTXZ2tpYvX36tp9plSkpKtHfvXlVWViomJkaZmZm67777lJyc7Nvm4sWLKioq0p49e9TU1KTs7GzNnz9fTqfT3MRDrDN9WLFihcrLy/3qpk+froULF17r6bYrLAJoz549Kioq0oIFCzRixAj94Q9/UH5+vtatW6d+/fqZnt41df311+vnP/+5bzkiouefxDY2NiotLU1Tp05VQUFBm/Vvvvmm3n77bS1atEhJSUl69dVXlZ+fr7Vr1yomJsbAjLtGR32QpLFjxyo3N9e3HBUVFt/inVZeXq4ZM2Zo+PDhunTpkl5++WU99dRTWrt2rfr06SNJ2rRpk8rKyvSjH/1I8fHxKiws1Jo1a/SLX/zC8OxDpzN9kKRp06Zp7ty5vuXu9v0QFkfn1q1bNW3aNN12222SpAULFqisrEw7d+7Ud7/7XbOTu8YiIiJ61G9ynZGTk6OcnJyA6zwej7Zt26bZs2dr/PjxkqTFixdrwYIF2rdvn2655ZZrOdUudaU+tIqKiurRx8flZ3OLFi3S/PnzVVFRoRtuuEH19fXasWOHlixZolGjRkmScnNz9dhjj+nw4cPKzMw0Me2Q66gPrWJjY7v18dDtA6i5uVkVFRV+QRMREaHRo0fr8OHD5iZmSFVVlX74wx8qOjpamZmZuvfee5WYmGh6WsacOXNGLpdLY8aM8Y3Fx8crIyNDhw8f7lEB1Bnl5eWaP3++7Ha7Ro0apXnz5snhcJieVpepr6+XJCUkJEiSKioqdOnSJY0ePdq3TUpKihITE3tUAF3u8j60Ki0tVWlpqZxOp8aNG6c5c+YoNjbWxBQD6vYBVFtbq5aWljYp7nQ6dfr0aTOTMmTEiBHKzc1VcnKyqqurtXnzZj3xxBNas2aN4uLiTE/PCJfLJUltLsX269fPt663GDt2rCZOnKikpCRVVVXp5Zdf1tNPP638/Pweeam2paVFv/vd7zRy5EgNHTpUkvd4iIqK8n0oW6uefDwE6oMk3XrrrUpMTNSAAQN0/PhxvfTSSzp9+rSWLl1qcLb+un0A4f99/fJLamqqL5A+/PBDTZ061eDM0B18/Wxv6NChSk1N1SOPPKJPPvnE74ygpygsLNTJkye1cuVK01Mxqr0+TJ8+3ffvoUOHqn///lq5cqWqqqo0ePDgaz3NgLr9r0V9+/ZVREREm99eXC5Xt762eS3Y7XYlJyerqqrK9FSMaT0Gampq/MZramp6/fExaNAgORyOHnl8FBYWqqysTE8++aQGDhzoG3c6nWpubm7zUdQ99Xhorw+BtL5quDsdD90+gKKiopSenq6DBw/6xlpaWnTw4MEeez23sxoaGlRVVdUjv7E6KykpSU6nUx9//LFvrL6+Xp9//nmvPz7OnTun8+fPq3///qanEjIej0eFhYXau3evnnjiCSUlJfmtT09PV2RkpN/xcPr0aZ09e7ZHHQ8d9SGQY8eOSVK3Oh7C4hLczJkztXHjRqWnpysjI0Pbtm1TY2OjpkyZYnpq11RRUZFuuukmJSYmqrq6WsXFxYqIiNCtt95qempdqjVoW505c0bHjh1TQkKCEhMTdccdd+j111/XkCFDlJSUpFdeeUX9+/f3vSqup7hSHxISEvTaa69p4sSJcjqd+vLLL/X73/9egwcPVnZ2tsFZh1ZhYaF2796tZcuWKS4uzndlJD4+XjExMYqPj9fUqVNVVFSkhIQExcfH6/nnn1dmZmaPCqCO+lBVVaXdu3frxhtvVEJCgk6cOKFNmzbpG9/4hlJTU81O/mvC5uMY3nnnHW3ZskUul0tpaWn6wQ9+oBEjRpie1jW1bt06ffrpp6qrq1Pfvn2VlZWlefPmdZvruV3lk08+UV5eXpvxyZMna9GiRb43ov7xj39UfX29srKy9PDDD/u9Ka8nuFIfFixYoNWrV+vo0aNyu90aMGCAxowZo7lz5/aoM+R77rkn4Hhubq7vF9LWN6J+8MEHam5u7pFvRO2oD2fPntW//uu/6uTJk2psbNTAgQM1YcIEzZ49W/Hx8dd4tu0LmwACAPQs3f45IABAz0QAAQCMIIAAAEYQQAAAIwggAIARBBAAwAgCCABgBAEEADCCAAIAGEEAAQCMIIAAAEb8L9Wg8qnvXMGhAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test Accuracy: 97.05%\n",
            "\n",
            "Confusion Matrix:\n",
            "[[ 974    1    1    0    0    1    2    1    0    0]\n",
            " [   0 1133    2    0    0    0    0    0    0    0]\n",
            " [  10    9  996    2    0    0    0   13    2    0]\n",
            " [   0    2    4  976    1   13    1    7    3    3]\n",
            " [   1    6    0    0  950    0    4    2    0   19]\n",
            " [   6    1    0   11    2  859    5    1    3    4]\n",
            " [   5    3    0    0    3    3  944    0    0    0]\n",
            " [   0   21    5    0    1    0    0  991    0   10]\n",
            " [   8    2    4   16    8   11    3    4  914    4]\n",
            " [   4    5    2    8    9    2    1    8    2  968]]\n"
          ]
        }
      ],
      "source": [
        "# Importing required libraries\n",
        "import numpy as np\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix\n",
        "import struct\n",
        "import matplotlib.pyplot as plt\n",
        "from matplotlib import style\n",
        "style.use('ggplot')\n",
        "\n",
        "# Function to load MNIST image files\n",
        "def load_images(filename):\n",
        "    with open(filename, 'rb') as f:\n",
        "        # Read magic number, number of images, rows, and columns\n",
        "        magic, num, rows, cols = struct.unpack(\">IIII\", f.read(16))\n",
        "        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)\n",
        "    return images\n",
        "\n",
        "# Function to load MNIST label files\n",
        "def load_labels(filename):\n",
        "    with open(filename, 'rb') as f:\n",
        "        # Read magic number and number of items\n",
        "        magic, num = struct.unpack(\">II\", f.read(8))\n",
        "        labels = np.fromfile(f, dtype=np.uint8)\n",
        "    return labels\n",
        "\n",
        "# Paths to your downloaded dataset files in Colab\n",
        "train_images_path = '/content/train-images-idx3-ubyte'\n",
        "train_labels_path = '/content/train-labels-idx1-ubyte'\n",
        "test_images_path = '/content/t10k-images-idx3-ubyte'\n",
        "test_labels_path = '/content/t10k-labels-idx1-ubyte'\n",
        "\n",
        "# Load training and testing data\n",
        "print(\"Loading training data...\")\n",
        "train_images = load_images(train_images_path)\n",
        "train_labels = load_labels(train_labels_path)\n",
        "\n",
        "print(\"Loading testing data...\")\n",
        "test_images = load_images(test_images_path)\n",
        "test_labels = load_labels(test_labels_path)\n",
        "\n",
        "# Check the shape of loaded data\n",
        "print(f\"Training images shape: {train_images.shape}\")\n",
        "print(f\"Training labels shape: {train_labels.shape}\")\n",
        "print(f\"Testing images shape: {test_images.shape}\")\n",
        "print(f\"Testing labels shape: {test_labels.shape}\")\n",
        "\n",
        "# Step 6: Example - Visualizing a sample from training data\n",
        "plt.imshow(train_images[0].reshape(28, 28), cmap='gray')\n",
        "plt.title(f\"Label: {train_labels[0]}\")\n",
        "plt.show()\n",
        "\n",
        "# Simple KNeighborsClassifier implementation\n",
        "knn = KNeighborsClassifier(n_neighbors=3)\n",
        "\n",
        "# Train the model on the training data\n",
        "knn.fit(train_images, train_labels)\n",
        "\n",
        "# Predict on the test data\n",
        "test_predictions = knn.predict(test_images)\n",
        "\n",
        "# Calculate accuracy\n",
        "test_accuracy = accuracy_score(test_labels, test_predictions)\n",
        "print(f\"Test Accuracy: {test_accuracy * 100:.2f}%\")\n",
        "\n",
        "# Confusion Matrix\n",
        "print(\"\\nConfusion Matrix:\")\n",
        "print(confusion_matrix(test_labels, test_predictions))\n"
      ]
    }
  ]
}
