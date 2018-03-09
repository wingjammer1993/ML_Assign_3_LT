import numpy as np
import matplotlib.pyplot as plt


def generate_concept_rect():
    a = np.random.rand()*100
    b = np.random.rand()*100
    c = np.random.rand()*100
    d = np.random.rand()*100
    return (a, b), (c, d), (a, d), (c, b)


def plot_rectangle(vtx1, vtx2, vtx3, vtx4, marker):
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.plot([vtx1[0], vtx2[0], vtx3[0], vtx4[0]], [vtx1[-1], vtx2[-1], vtx3[-1], vtx4[-1]], marker)


def generate_training_data(num_samples):
    sample_list = []
    for k in range(0, num_samples):
        a = np.random.rand()*100
        b = np.random.rand()*100
        sample_list.append((a, b))
    return sample_list


def generate_normal_training_data(num_samples):
    sample_list = []
    for k in range(0, num_samples):
        a = 50*np.random.randn() + 25
        b = 50*np.random.randn() + 25
        sample_list.append((a, b))
    return sample_list


def label_data_concept(concept, data):
    labeled_training = {}
    for ex in data:
        if concept[0] <= ex[0] <= concept[2] or concept[0] >= ex[0] >= concept[2]:
            if concept[1] <= ex[-1] <= concept[3] or concept[1] >= ex[-1] >= concept[3]:
                labeled_training[ex] = 1
            else:
                labeled_training[ex] = 0
        else:
            labeled_training[ex] = 0
    return labeled_training


def plot_labelled_training(label_dict):
    x_p = []
    y_p = []
    x_n = []
    y_n = []
    for label in label_dict:
        if label_dict[label] == 1:
            x_p.append(label[0])
            y_p.append(label[-1])
        else:
            x_n.append(label[0])
            y_n.append(label[-1])
    plt.plot(x_p, y_p, 'r.')
    plt.plot(x_n, y_n, 'b.')


def give_hypothesis(label_dict):
    x_p = []
    y_p = []
    for label in label_dict:
        if label_dict[label] == 1:
            x_p.append(label[0])
            y_p.append(label[-1])
    if len(x_p) > 0 or len(y_p) > 0:
        x_min = min(x_p)
        x_max = max(x_p)
        y_min = min(y_p)
        y_max = max(y_p)
        print(x_min, x_max, y_min, y_max)
        return (x_min, y_min), (x_max, y_max), (x_min, y_max), (x_max, y_min)
    else:
        return (0, 0), (0, 0), (0, 0), (0, 0)


def calculate_gen_error(concept, hypothesis):
    count = 0
    for p in concept:
        if concept[p] != hypothesis[p]:
            count += 1
    return count/len(concept)


if __name__ == "__main__":

    # Set up
    m = 100
    vx1, vx2, vx3, vx4 = generate_concept_rect()
    plot_rectangle(vx1, vx2, vx3, vx4, 'yo')
    training_data = generate_training_data(m)
    dict_label = label_data_concept((vx1[0], vx1[-1], vx2[0], vx2[-1]), training_data)
    plot_labelled_training(dict_label)
    vh1, vh2, vh3, vh4 = give_hypothesis(dict_label)
    if vh1 != 'error' or vh2 != 'error' or vh3 != 'error' or vh4 != 'error':
        plot_rectangle(vh1, vh2, vh3, vh4, 'g.')
        # Training is over, hypothesis is created, validate the results
        validation_data = generate_training_data(1000)
        concept_label = label_data_concept((vx1[0], vx1[-1], vx2[0], vx2[-1]), validation_data)
        hypothesis_label = label_data_concept((vh1[0], vh1[-1], vh2[0], vh2[-1]), validation_data)
        error = calculate_gen_error(concept_label, hypothesis_label)
        print(error)
    plt.show()
    plt.close()
    print(vx1, vx2, vx3, vx4)

    # Part 1

    m = 100
    gen_error = []
    for i in range(0, 500):
        vx1, vx2, vx3, vx4 = generate_concept_rect()
        training_data = generate_training_data(m)
        dict_label = label_data_concept((vx1[0], vx1[-1], vx2[0], vx2[-1]), training_data)
        vh1, vh2, vh3, vh4 = give_hypothesis(dict_label)
        if vh1 != 'error' or vh2 != 'error' or vh3 != 'error' or vh4 != 'error':
            # Training is over, hypothesis is created, validate the results
            validation_data = generate_training_data(1000)
            concept_label = label_data_concept((vx1[0], vx1[-1], vx2[0], vx2[-1]), validation_data)
            hypothesis_label = label_data_concept((vh1[0], vh1[-1], vh2[0], vh2[-1]), validation_data)
            error = calculate_gen_error(concept_label, hypothesis_label)
            gen_error.append(error)

    percentile = np.percentile(gen_error, 95)
    print('95th percentile of generalization error is {}'.format(percentile))

    # Part 2

    m_list = [250, 500, 1000, 1250, 1500]
    gen_error_m = {}
    gen_err = []
    for m in m_list:
        for i in range(0, 100):
            vx1, vx2, vx3, vx4 = generate_concept_rect()
            training_data = generate_training_data(m)
            dict_label = label_data_concept((vx1[0], vx1[-1], vx2[0], vx2[-1]), training_data)
            vh1, vh2, vh3, vh4 = give_hypothesis(dict_label)
            if vh1 != 'error' or vh2 != 'error' or vh3 != 'error' or vh4 != 'error':
                # Training is over, hypothesis is created, validate the results
                validation_data = generate_training_data(1000)
                concept_label = label_data_concept((vx1[0], vx1[-1], vx2[0], vx2[-1]), validation_data)
                hypothesis_label = label_data_concept((vh1[0], vh1[-1], vh2[0], vh2[-1]), validation_data)
                error = calculate_gen_error(concept_label, hypothesis_label)
                gen_err.append(error)
        percentile = np.percentile(gen_err, 95)
        gen_error_m[m] = np.log(percentile)
    print(gen_error_m)
    theory_gen = [np.log(17.52/m) for m in m_list]
    plt.figure()
    m_list = [np.log(m) for m in m_list]
    plt.plot(m_list, theory_gen, label='theoretical bound')
    plt.plot(m_list, gen_error_m.values(), label='empirical bound')
    plt.legend()
    plt.show()
    plt.close()

    # Part 3

    m_list_2 = [250, 500, 1000, 1250, 1500]
    gen_error_m_2 = {}
    gen_err_2 = []
    for m in m_list_2:
        for i in range(0, 100):
            vx1, vx2, vx3, vx4 = generate_concept_rect()
            training_data = generate_normal_training_data(m)
            dict_label = label_data_concept((vx1[0], vx1[-1], vx2[0], vx2[-1]), training_data)
            vh1, vh2, vh3, vh4 = give_hypothesis(dict_label)
            if vh1 != 'error' or vh2 != 'error' or vh3 != 'error' or vh4 != 'error':
                # Training is over, hypothesis is created, validate the results
                validation_data = generate_normal_training_data(1000)
                concept_label = label_data_concept((vx1[0], vx1[-1], vx2[0], vx2[-1]), validation_data)
                hypothesis_label = label_data_concept((vh1[0], vh1[-1], vh2[0], vh2[-1]), validation_data)
                error = calculate_gen_error(concept_label, hypothesis_label)
                gen_err_2.append(error)
        percentile = np.percentile(gen_err_2, 95)
        gen_error_m_2[m] = np.log(percentile)
    print(gen_error_m_2)
    theory_gen = [np.log(17.52/m) for m in m_list_2]
    m_list_2 = [np.log(m) for m in m_list_2]
    plt.figure()
    plt.plot(m_list_2, theory_gen, label='theoretical bound')
    plt.plot(m_list_2, gen_error_m_2.values(), label='empirical bound')
    plt.legend()
    plt.show()
    plt.close()



