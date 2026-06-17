(function () {
  const dataNode = document.getElementById("dashboard-data");
  const initialPayload = parseJson(dataNode ? dataNode.textContent : "{}");
  renderDailyChart(initialPayload.daily || []);
  bindFilters();
  bindBreakdown();
  bindAgent();
})();

const FILTER_DEFAULTS = {
  period: "this_month",
  start_date: "",
  end_date: "",
  category: "",
  direction: "expense",
  group_by: "category",
};

function bindFilters() {
  const form = document.getElementById("dashboard-filters");
  if (!form) {
    return;
  }

  const debouncedLoad = debounce(loadDashboard, 350);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    loadDashboard();
  });

  form.querySelectorAll("select, input").forEach((control) => {
    const isDate = control.name === "start_date" || control.name === "end_date";
    control.addEventListener(isDate ? "input" : "change", () => {
      if (isDate) {
        const period = form.elements.period;
        if (period) {
          period.value = "custom";
        }
        debouncedLoad();
      } else {
        loadDashboard();
      }
    });
  });

  const reset = document.getElementById("filters-reset");
  if (reset) {
    reset.addEventListener("click", () => {
      Object.entries(FILTER_DEFAULTS).forEach(([name, value]) => {
        const control = form.elements[name];
        if (control) {
          control.value = value;
        }
      });
      loadDashboard();
    });
  }
}

function bindBreakdown() {
  const list = document.getElementById("breakdown-list");
  if (!list) {
    return;
  }
  list.addEventListener("click", (event) => {
    const item = event.target.closest("[data-breakdown-item]");
    if (!item) {
      return;
    }
    loadTransactions(item.dataset.bucket || "");
  });
}

function bindAgent() {
  const form = document.getElementById("agent-form");
  if (!form) {
    return;
  }

  const submitButton = form.querySelector("button[type='submit']");
  const textarea = form.querySelector("textarea[name='question']");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const answer = document.getElementById("agent-answer");
    const question = String(new FormData(form).get("question") || "").trim();
    const filtersForm = document.getElementById("dashboard-filters");
    const familyId = filtersForm
      ? String(new FormData(filtersForm).get("family_id") || "")
      : "";
    if (!question || !familyId) {
      setAgentAnswer("Нужен вопрос и выбранная семья.");
      return;
    }

    setAgentAnswer("Агент считает...");
    setLoading(submitButton, true);
    try {
      const response = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ family_id: familyId, question }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "agent failed");
      }
      setAgentAnswer(payload.answer || "Агент не вернул ответ.");
    } catch (error) {
      setAgentAnswer(`Ошибка агента: ${error.message}`);
    } finally {
      setLoading(submitButton, false);
    }

    if (answer) {
      answer.scrollIntoView({ block: "nearest" });
    }
  });

  if (textarea) {
    textarea.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
      }
    });
  }

  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      if (textarea) {
        textarea.value = button.dataset.prompt || "";
      }
      form.requestSubmit();
    });
  });
}

async function loadDashboard() {
  const form = document.getElementById("dashboard-filters");
  if (!form) {
    return;
  }

  const params = new URLSearchParams(new FormData(form));
  const url = `/api/dashboard?${params.toString()}`;
  try {
    setBusy(true);
    const response = await fetch(url);
    const dashboard = await response.json();
    if (!response.ok) {
      throw new Error(dashboard.error || "dashboard failed");
    }
    setFilterError("");
    renderDashboard(dashboard);
    window.history.replaceState(null, "", `/?${params.toString()}`);
  } catch (error) {
    setFilterError(`Ошибка загрузки: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function loadTransactions(bucket) {
  const form = document.getElementById("dashboard-filters");
  if (!form) {
    return;
  }
  const params = new URLSearchParams(new FormData(form));
  if (bucket) {
    params.set("bucket", bucket);
  }
  const response = await fetch(`/api/transactions?${params.toString()}`);
  const detail = await response.json();
  if (!response.ok) {
    setFilterError(detail.error || "Не удалось загрузить операции.");
    return;
  }
  setFilterError("");
  renderTransactions(detail.transactions || [], detail.title || "Операции");
}

function renderDashboard(dashboard) {
  setText("context-period", dashboard.period_label);
  setText("context-generated", dashboard.generated_at);
  setText("context-family", dashboard.selected_family_name || "");
  renderMetrics(dashboard.metrics || []);
  renderDailyChart(dashboard.daily_points || []);
  renderBreakdown(dashboard.categories || []);
  renderBudgets(dashboard.budgets || []);
  renderSubscriptions(dashboard.subscriptions || [], dashboard.subscriptions_total || "0 ₽");
  renderGoal(dashboard.goal);
  renderTransactions(dashboard.recent_transactions || [], "Последние движения");
  updateFilterValues(dashboard.filters || {});
}

function renderMetrics(metrics) {
  const root = document.getElementById("metric-grid");
  if (!root) {
    return;
  }
  root.textContent = "";
  metrics.forEach((metric) => {
    const card = el("article", `metric metric--${metric.tone || "neutral"}`);
    card.append(
      el("span", "", metric.label),
      el("strong", "", metric.value),
      el("small", "", metric.detail),
      el("em", "", metric.trend),
    );
    root.append(card);
  });
}

function renderBreakdown(items) {
  const root = document.getElementById("breakdown-list");
  if (!root) {
    return;
  }
  root.textContent = "";
  if (!items.length) {
    root.append(el("p", "muted", "Данных по выбранным параметрам нет."));
    return;
  }
  items.forEach((item) => {
    const row = el("article", "row-item row-item--clickable");
    row.dataset.breakdownItem = "true";
    row.dataset.bucket = item.bucket || "";
    row.dataset.groupBy = item.group_by || "";

    const main = el("div", "row-item__main");
    main.append(el("strong", "", item.label), el("span", "", `${item.count} операций · ${item.pct}%`));

    const progress = el("div", "progress");
    const bar = el("span");
    bar.style.width = `${item.bar_pct || 0}%`;
    progress.append(bar);

    row.append(main, el("div", "row-item__amount", item.total), progress);
    root.append(row);
  });
}

function renderBudgets(budgets) {
  const root = document.getElementById("budget-list");
  if (!root) {
    return;
  }
  root.textContent = "";
  if (!budgets.length) {
    root.append(el("p", "muted", "Бюджеты ещё не настроены."));
    return;
  }
  budgets.forEach((budget) => {
    const row = el("article", `row-item row-item--status-${budget.status}`);
    const main = el("div", "row-item__main");
    main.append(el("strong", "", budget.label), el("span", "", `${budget.spent} из ${budget.limit}`));
    const pill = el("div", `status-pill status-pill--${budget.status}`, `${budget.status_label} · ${budget.pct}%`);
    const progress = el("div", `progress progress--${budget.status}`);
    const bar = el("span");
    bar.style.width = `${budget.bar_pct || 0}%`;
    progress.append(bar);
    row.append(main, pill, progress);
    root.append(row);
  });
}

function renderSubscriptions(subscriptions, total) {
  setText("subscriptions-total", `${total} / мес`);
  const root = document.getElementById("subscription-list");
  if (!root) {
    return;
  }
  root.textContent = "";
  if (!subscriptions.length) {
    root.append(el("p", "muted", "Регулярных списаний за последний год не найдено."));
    return;
  }
  subscriptions.forEach((sub) => {
    const row = el("article", "row-item");
    const main = el("div", "row-item__main");
    main.append(
      el("strong", "", sub.merchant),
      el("span", "", `${sub.category} · ${sub.cadence} · последнее ${sub.last_seen}`),
    );
    const amount = el("div", "row-item__amount", sub.monthly);
    amount.append(el("small", "", sub.last_amount));
    row.append(main, amount);
    root.append(row);
  });
}

function renderGoal(goal) {
  const root = document.getElementById("goal-panel");
  if (!root) {
    return;
  }
  const heading = root.querySelector(".section-heading");
  root.textContent = "";
  if (heading) {
    root.append(heading);
  }
  if (!goal) {
    root.append(el("p", "muted", "Цель накопления пока не задана."));
    return;
  }

  const meter = el("div", "goal-meter");
  const left = el("div");
  left.append(el("span", "", "Накоплено"), el("strong", "", goal.saved), el("small", "", `из ${goal.target}`));
  meter.append(left, el("b", "", `${goal.pct}%`));

  const progress = el("div", "progress progress--goal");
  const bar = el("span");
  bar.style.width = `${goal.bar_pct || 0}%`;
  progress.append(bar);

  const facts = el("dl", "facts");
  facts.append(fact("Осталось", goal.remaining), fact("Дата", goal.target_date));
  if (goal.monthly_needed) {
    facts.append(fact("В месяц", goal.monthly_needed));
  }

  root.append(meter, progress, facts);
}

function renderTransactions(transactions, title) {
  setText("transaction-title", title);
  const root = document.getElementById("transaction-list");
  if (!root) {
    return;
  }
  root.textContent = "";
  if (!transactions.length) {
    root.append(el("p", "muted", "Операций пока нет."));
    return;
  }
  transactions.forEach((tx) => {
    const row = el("article", "transaction");
    const main = el("div");
    main.append(el("strong", "", tx.merchant), el("span", "", `${tx.category} · ${tx.direction}`));
    row.append(
      el("time", "", tx.occurred_at),
      main,
      el("b", `amount amount--${tx.tone || "neutral"}`, tx.amount),
    );
    root.append(row);
  });
}

function renderDailyChart(points) {
  const root = document.getElementById("daily-chart");
  if (!root) {
    return;
  }

  const maxValue = points.reduce(
    (max, point) => Math.max(max, Number(point.income || 0), Number(point.expense || 0)),
    0,
  );

  root.textContent = "";
  if (!points.length || maxValue <= 0) {
    root.append(el("div", "chart__empty", "Нет операций за период"));
    return;
  }

  root.style.setProperty("--chart-count", String(points.length));
  points.forEach((point) => {
    const day = el("div", "chart-day");
    day.dataset.label = String(point.label);

    const income = el("div", "chart-bar chart-bar--income");
    income.style.height = `${heightFor(point.income, maxValue)}px`;
    income.title = `Доходы: ${formatRub(point.income)}`;

    const expense = el("div", "chart-bar chart-bar--expense");
    expense.style.height = `${heightFor(point.expense, maxValue)}px`;
    expense.title = `Расходы: ${formatRub(point.expense)}`;

    day.append(income, expense);
    root.append(day);
  });
}

function updateFilterValues(filters) {
  const form = document.getElementById("dashboard-filters");
  if (!form) {
    return;
  }
  ["period", "start_date", "end_date", "category", "direction", "group_by"].forEach((name) => {
    const control = form.elements[name];
    if (control && filters[name] !== undefined) {
      control.value = filters[name];
    }
  });
}

function fact(label, value) {
  const item = el("div");
  item.append(el("dt", "", label), el("dd", "", value));
  return item;
}

function setAgentAnswer(text) {
  const answer = document.getElementById("agent-answer");
  if (!answer) {
    return;
  }
  answer.textContent = text;
}

function setFilterError(text) {
  const node = document.getElementById("filter-error");
  if (!node) {
    return;
  }
  node.textContent = text || "";
  node.hidden = !text;
}

function setBusy(isBusy) {
  const form = document.getElementById("dashboard-filters");
  if (!form) {
    return;
  }
  form.classList.toggle("is-busy", isBusy);
}

function setLoading(button, isLoading) {
  if (!button) {
    return;
  }
  button.classList.toggle("is-loading", isLoading);
  button.disabled = isLoading;
}

function debounce(fn, delay) {
  let timer = null;
  return function debounced(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

function heightFor(value, maxValue) {
  const number = Number(value || 0);
  if (number <= 0) {
    return 2;
  }
  return Math.max(2, Math.round((number / maxValue) * 220));
}

function formatRub(value) {
  return `${Math.round(Number(value || 0)).toLocaleString("ru-RU")} ₽`;
}

function parseJson(value) {
  try {
    return JSON.parse(value || "{}");
  } catch {
    return {};
  }
}

function setText(id, text) {
  const node = document.getElementById(id);
  if (node) {
    node.textContent = text;
  }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}
