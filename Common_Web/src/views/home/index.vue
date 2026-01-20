<template>
  <div class="c-flex c-h100p c-w100p">
    <section class="c-w300 c-p10" style="border-right:1px solid gray">历史记录</section>
    <section class="c-flex-1 c-flex c-flex-column">
      <section class="c-flex-1 c-p10">
        <h1 class="c-text-center">您好，我是多模态聊天机器人，请输入您的问题！</h1>
        <div v-for="(item,index) in store" :key="index">
          <div v-show="item.input" class="c-text-end">{{ item.input }}</div>
          <n-spin :show="loading && index===store.length-1">
            <div v-show="item.output" class="c-text-start">{{ item.output }}</div>
          </n-spin>
        </div>
      </section>
      <section style="border-top:1px solid gray">
        <n-input
          v-model:value="text"
          type="textarea"
          placeholder="请输入"
          class="c-m10"
        />
        <div class="c-flex c-m10">
          <n-button type="info" class="c-mr10">深度思考</n-button>
          <n-button type="info" class="c-mr10">图片</n-button>
          <n-button type="info" class="c-mr10">语音</n-button>
          <n-button type="info" class="c-mr10">视频</n-button>
          <n-button type="info" class="c-mr10">上传文件</n-button>
          <n-button type="info" class="c-mr10" @click="testHandle">测试</n-button>
        </div>
      </section>
    </section>
  </div>
</template>

<script setup lang="ts">
import {shallowRef, ref} from "vue";
import {test} from './request.ts'

type Store = {
  input: string
  output: string
}[]
const store = ref<Store>([])
const text = shallowRef('')
const loading = shallowRef(false)

function testHandle() {
  loading.value = true
  store.value.push({input: text.value, output: ''})
  test({content: text.value}).then(res => {
    store.value[store.value.length - 1] = {input: text.value, output: res.data}
    text.value = ''
  }).finally(() => {
    loading.value = false
  })
}
</script>

<style scoped>

</style>
