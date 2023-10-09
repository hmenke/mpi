// Copyright (c) 2023 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Philipp Dumitrescu, Olivier Parcollet, Nils Wentzell

#pragma once
#include "mpi.hpp"
#include <algorithm>
#include <memory>

namespace mpi {

  template <class T>
  class shared_allocator {
    shared_communicator shm = communicator{}.split_shared();
    std::shared_ptr<std::vector<shared_window<T>>> blocks = std::make_shared<std::vector<shared_window<T>>>();
  public:
    using value_type = T;

    shared_allocator() = default;
    explicit shared_allocator(shared_communicator const &shm) noexcept : shm{shm} {}

    [[nodiscard]] shared_communicator get_communicator() { return shm; }

    [[nodiscard]] auto get_window(T *p) {
      return std::find_if(blocks->begin(), blocks->end(), [p](shared_window<T> const &win) {
        return win.base(0) == p;
      });
    }

    [[nodiscard]] T* allocate(std::size_t n) {
      shared_window<T> &win = blocks->emplace_back(shm, shm.rank() == 0 ? n : 0);
      return win.base(0);
    }

    void deallocate(T *p, std::size_t) {
      blocks->erase(std::remove_if(blocks->begin(), blocks->end(), [p](shared_window<T> const &win) {
        return win.base(0) == p;
      }), blocks->end());
    }
  };

} // namespace mpi
